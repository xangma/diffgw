#!/usr/bin/env ruby
# Local-only structural renderer; it never invokes SpECTRE or SSH.
require "yaml"
require "json"
require "digest"
require "fileutils"

template, output_dir, record = ARGV
abort "usage: TEMPLATE OUTPUT_DIR RECORD" unless template && output_dir && record
abort "output exists" if File.exist?(output_dir) || File.exist?(record)
docs = YAML.load_stream(File.read(template, mode: "r"))
abort "expected metadata and options documents" unless docs.length == 2
metadata, options = docs
abort "wrong executable" unless metadata["Executable"] == "EvolveGhSingleBlackHole"
abort "missing Kerr-Schild data" unless options.dig("InitialData", "GeneralizedHarmonic(KerrSchild)")

phases = options.fetch("PhaseChangeAndTriggers")
matches = phases.flat_map { |entry| entry.fetch("PhaseChanges").filter_map { |x| x["CheckpointAndExitAfterWallclock"] } }
abort "checkpoint option must be unique" unless matches.length == 1
checkpoint = matches.fetch(0)
abort "unexpected checkpoint setting" unless checkpoint["WallclockHours"].to_s == "None"
checkpoint["WallclockHours"] = 0.0
phase_entries = phases.select { |entry| entry.dig("Trigger", "Slabs", "EvenlySpaced", "Interval") == 100 }
abort "checkpoint sync trigger must be unique" unless phase_entries.length == 1
phase_entries.fetch(0)["Trigger"] = {"TimeCompares" => {"Comparison" => "GreaterThanOrEqualTo", "Value" => 0.001}}

slab_events = options.fetch("EventsAndTriggersAtSlabs")
completion = slab_events.select { |entry| entry.fetch("Events") == ["Completion"] }
abort "completion trigger must be unique" unless completion.length == 1
completion.fetch(0)["Trigger"] = {"TimeCompares" => {"Comparison" => "GreaterThanOrEqualTo", "Value" => 0.002}}

FileUtils.mkdir_p(File.join(output_dir, "Segment_0000"))
input = File.join(output_dir, "Segment_0000", "KerrSchild.yaml")
File.write(input, YAML.dump_stream(metadata, options))
payload = {
  "template" => template, "template_sha256" => Digest::SHA256.file(template).hexdigest,
  "input" => input, "input_sha256" => Digest::SHA256.file(input).hexdigest,
  "num_procs" => 4,
  "first_segment_command" => ["EvolveGhSingleBlackHole", "+p4", "--input-file", input],
  "validation_command" => ["EvolveGhSingleBlackHole", "+p4", "--input-file", input, "--check-options"],
  "restart_command" => ["EvolveGhSingleBlackHole", "+p4", "+restart", "../Segment_0000/Checkpoints/Checkpoint_0000"],
  "restart_overlay" => {"phase_type" => "CheckpointAndExitAfterWallclock", "WallclockHours" => "None", "input_file_argument" => false},
  "checkpoint_trigger" => {"comparison" => "GreaterThanOrEqualTo", "time" => 0.001, "WallclockHours" => 0.0},
  "completion_trigger" => {"comparison" => "GreaterThanOrEqualTo", "time" => 0.002},
  "launched" => false,
  "scope" => "local render only; parent must validate and launch"
}
File.write(record, JSON.pretty_generate(payload) + "\n")
puts JSON.pretty_generate(payload)
