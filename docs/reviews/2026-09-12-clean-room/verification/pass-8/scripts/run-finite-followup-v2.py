#!/usr/bin/env python3
"""Run the finite NR prerequisites on len, sequentially, stopping on failure.

Reuses hash-locked, accepted controlled data and preparation records. It never
queues the full waveform campaign or retries an existing stage.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

ROOT = Path("/home/xangma/simulations/torchwave-nr/2026-09-12-pilot")
PROVENANCE = ROOT / "provenance"
IMAGE = "sha256:0ba0f3bb70b90e83c9b50316ecd37f1b55af4537160eafbe3392a6f67c4e9d77"
PLAN = PROVENANCE / "finite-followup-v2-plan.json"
STATE = PROVENANCE / "finite-followup-v2-state.json"
TOTAL_LIMIT = 48000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted(metrics, expected):
    assert metrics["exit_code"] == metrics["expected_exit_code"] == expected, metrics
    assert metrics["accepted_exit"] is True and metrics["watchdog_reason"] is None, metrics
    counters = dict(line.split() for line in metrics["cgroup_memory_events"].splitlines())
    assert all(int(counters[key]) == 0 for key in ("oom", "oom_kill", "oom_group_kill")), counters


def host_path(container_path):
    path = Path(container_path)
    assert path.is_absolute() and ".." not in path.parts
    return ROOT / path.relative_to("/campaign")


class Runner:
    def __init__(self):
        self.started = time.monotonic()
        self.stopping = False
        self.owned = None
        self.plan = json.loads(PLAN.read_text())
        job = json.loads((PROVENANCE / "finite-followup-v2-job.json").read_text())
        assert job["plan_sha256"] == sha256(PLAN), "Plan changed after launch"
        assert job["host"] == "len" and job["cwd"] == str(ROOT)
        assert job["command"] == ["/usr/bin/python3", "-u", str(Path(__file__).resolve())]
        self.state = {"host": "len", "pid": os.getpid(), "cwd": str(ROOT),
                      "started_utc": self.now(), "status": "starting", "completed": [],
                      "plan_sha256": sha256(PLAN), "current_owned_service": None}

    @staticmethod
    def now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def save(self, **updates):
        self.state.update(updates, updated_utc=self.now(), current_owned_service=self.owned)
        temporary = STATE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(STATE)
        print(json.dumps(self.state), flush=True)

    def interrupt(self, signum, frame):
        self.stopping = True

    def check_stop(self):
        if self.stopping:
            raise RuntimeError("Stop requested")
        if time.monotonic() - self.started > TOTAL_LIMIT:
            raise RuntimeError("Finite continuation time budget exhausted")

    def verify_helpers(self):
        assert sha256(PLAN) == self.state["plan_sha256"], "Plan changed while running"
        for name, digest in self.plan["helper_sha256"].items():
            assert Path(name).name == name
            assert sha256(ROOT / "scripts" / name) == digest, name

    def wait_stage(self, name, expected, seconds):
        deadline = time.monotonic() + seconds
        missing_terminal = 0
        while True:
            self.check_stop()
            assert time.monotonic() < deadline, f"Timed out waiting for {name}"
            metrics_path = PROVENANCE / (name + "-metrics.json")
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text())
                except json.JSONDecodeError:
                    time.sleep(1)
                    continue
                accepted(metrics, expected)
                job = json.loads((PROVENANCE / (name + "-job.json")).read_text())
                assert metrics["command"] == job["command"]
                assert metrics["cwd"] == job["container_cwd"]
                assert (datetime.datetime.fromisoformat(metrics["started_utc"])
                        >= datetime.datetime.fromisoformat(job["requested_utc"]))
                return metrics
            result = subprocess.run(
                ["systemctl", "--user", "show", "torchwave-nr-" + name + ".service",
                 "--property=ActiveState", "--value"], capture_output=True,
                text=True, check=True, timeout=20)
            if result.stdout.strip() not in ("active", "activating", "deactivating"):
                missing_terminal += 1
                assert missing_terminal < 2, f"{name} stopped without metrics"
            else:
                missing_terminal = 0
            time.sleep(30)

    def resource_check(self, memory_gib):
        self.check_stop()
        meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(meminfo["MemAvailable"].split()[0]) * 1024
        assert available >= (memory_gib + 16) * 2**30, "Insufficient available memory; no stage launched"
        assert shutil.disk_usage(ROOT).free >= 80 * 2**30, "Less than 80GiB free; no stage launched"

    def launch(self, name, cwd, output, command, cpus=1, memory=4, seconds=180, expected=0):
        self.verify_helpers()
        self.resource_check(memory)
        assert not any((PROVENANCE / (name + suffix)).exists()
                       for suffix in ("-metrics.json", "-job.json"))
        if command[0] in ("EvolveGhBinaryBlackHole", "spectre"):
            workdir = host_path(cwd)
            assert not list(workdir.glob("*.h5")), "Existing simulation output"
            assert not (workdir / "Checkpoints").exists(), "Existing checkpoints"
        self.owned = "torchwave-nr-" + name + ".service"
        self.save(status="running", stage=name, command=command)
        launcher = ["python3", str(ROOT / "scripts/launch-stage.py"), name,
                    "--image", IMAGE, "--workdir", cwd, "--output-dir", output,
                    "--cpus", str(cpus), "--memory-gib", str(memory),
                    "--timeout-seconds", str(seconds), "--expect-exit", str(expected),
                    "--command-json", json.dumps(command)]
        subprocess.run(launcher, check=True, timeout=60)
        metrics = self.wait_stage(name, expected, seconds + 120)
        self.state["completed"].append({"stage": name, "wall_seconds": metrics["wall_seconds"],
                                       "metrics_sha256": sha256(PROVENANCE / (name + "-metrics.json"))})
        self.owned = None
        self.save(status="between_stages", stage=name)

    def verify_manifest(self, name, path_key, hash_key):
        manifest = json.loads((PROVENANCE / name).read_text())
        assert sha256(host_path(manifest[path_key])) == manifest[hash_key]
        return manifest

    def verify_completed_inputs(self):
        for relative, digest in self.plan["completed_input_sha256"].items():
            path = Path(relative)
            assert not path.is_absolute() and ".." not in path.parts
            (ROOT / path).resolve(strict=True).relative_to(ROOT.resolve())
            assert sha256(ROOT / path) == digest, relative

    def reuse_preparation(self, name):
        self.verify_completed_inputs()
        record = self.plan["completed_preparations"][name]
        metrics = self.wait_stage(name, 0, 60)
        job = json.loads((PROVENANCE / (name + "-job.json")).read_text())
        assert job["image_id"] == IMAGE
        assert job["command"] == record["command"]
        assert job["container_cwd"] == record["cwd"]
        assert job["cpus"] == 1 and job["memory_gib"] == 4
        self.state["completed"].append({"stage": name, "reused": True,
            "wall_seconds": metrics["wall_seconds"],
            "metrics_sha256": sha256(PROVENANCE / (name + "-metrics.json"))})
        self.save(status="between_stages", stage=name)

    def run(self):
        self.verify_helpers()
        self.verify_completed_inputs()
        assert self.plan["image_id"] == IMAGE
        proof = PROVENANCE / self.plan["single_hole_validation_file"]
        assert sha256(proof) == self.plan["single_hole_validation_sha256"]
        assert json.loads(proof.read_text())["validation_passed"] is True
        self.save(status="waiting", stage="controlled-id")
        self.wait_stage("controlled-id", 0, 14000)
        quality = json.loads((ROOT / "controlled-id/final-quality.json").read_text())
        assert quality["target_converged"] is True and quality["target_tolerance"] == 5e-5
        host_path(quality["run_dir"]).relative_to(ROOT / "controlled-id")

        self.reuse_preparation("p11-preparation-v2")
        self.verify_manifest("independent-p11-manifest.json", "rendered_initial_data", "rendered_initial_data_sha256")
        self.launch("p11-solve", "/campaign/independent-p11", "/campaign/independent-p11",
                    ["spectre", "schedule", "/campaign/independent-p11/InitialData.yaml",
                     "--no-schedule", "--num-procs", "16", "--submit"],
                    cpus=16, memory=160, seconds=13800)
        self.launch("p11-comparison", "/campaign/validation/independent-p11",
                    "/campaign/validation/independent-p11",
                    ["python", "/campaign/scripts/compare-independent-p11.py"])
        comparison = json.loads((PROVENANCE / "independent-p11-comparison.json").read_text())
        assert comparison["comparison_passed"] is True

        self.reuse_preparation("controlled-p9-lev0-preparation-v2")
        manifest = self.verify_manifest("controlled-p9-lev0-evolution-manifest.json",
                                        "rendered_input", "rendered_input_sha256")
        assert manifest["configuration"]["num_procs"] == 16
        assert manifest["configuration"]["final_time"] == 1.0
        assert manifest["configuration"]["checkpoint_time"] == 0.5
        run = "/campaign/controlled-p9-lev0-evolution"
        self.launch("controlled-p9-lev0-checkpoint", run + "/Segment_0000", run,
                    ["EvolveGhBinaryBlackHole", "+p16", "--input-file", run + "/Segment_0000/Inspiral.yaml"],
                    cpus=16, memory=96, seconds=6900, expected=2)
        self.launch("controlled-p9-lev0-restart-preparation",
                    "/campaign/preparation/controlled-p9-lev0-restart",
                    "/campaign/preparation/controlled-p9-lev0-restart",
                    ["python", "/campaign/scripts/prepare-smoke-restart.py", "--evolution-dir", run,
                     "--metrics", "/campaign/provenance/controlled-p9-lev0-checkpoint-metrics.json",
                     "--record", "/campaign/provenance/controlled-p9-lev0-restart-preparation.json",
                     "--num-procs", "16"])
        restart = json.loads((PROVENANCE / "controlled-p9-lev0-restart-preparation.json").read_text())
        assert sha256(host_path(restart["overlay"])) == restart["overlay_sha256"]
        assert restart["command"] == ["EvolveGhBinaryBlackHole", "+p16", "+restart",
                                       run + "/Segment_0000/Checkpoints/Checkpoint_0000"]
        self.launch("controlled-p9-lev0-restart", run + "/Segment_0001", run,
                    restart["command"], cpus=16, memory=96, seconds=6900)
        self.launch("controlled-p9-lev0-validation", "/campaign/validation/controlled-p9-lev0",
                    "/campaign/validation/controlled-p9-lev0",
                    ["python", "/campaign/scripts/validate-controlled-bbh-smoke.py"], seconds=300)
        result = json.loads((PROVENANCE / "controlled-p9-lev0-smoke-quality.json").read_text())
        assert result["validation_passed"] is True
        self.save(status="complete", stage="finite_checks_complete",
                  scope="P9/P11 initial-data check and 1M binary restart only; longer evolution remains unlaunched")


def main():
    assert socket.gethostname() == "len"
    assert not STATE.exists(), "Existing finite continuation requires explicit review; no automatic retry"
    runner = Runner()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, runner.interrupt)
    try:
        runner.run()
    except BaseException as error:
        runner.save(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if runner.owned:
            subprocess.run(["systemctl", "--user", "stop", runner.owned],
                           timeout=60, check=False)


if __name__ == "__main__":
    main()
