import os
import shutil
from pathlib import Path

def on_pre_build(config):
    """Ensure benchmark plot artifacts are mirrored into docs/artifacts before build."""
    root = Path(config["config_file_path"]).parent
    src_plots = root / "artifacts" / "benchmarks" / "plots"
    dst_plots = root / "docs" / "artifacts" / "benchmarks" / "plots"

    if src_plots.exists():
        dst_plots.mkdir(parents=True, exist_ok=True)
        for item in src_plots.iterdir():
            if item.is_file():
                target = dst_plots / item.name
                if not target.exists() or target.stat().st_mtime < item.stat().st_mtime:
                    shutil.copy2(item, target)


def on_page_markdown(markdown: str, page, config, files) -> str:
    """Normalize cross-repo relative links for MkDocs site generation."""
    # Mirror ../artifacts/ links to artifacts/ within docs
    markdown = markdown.replace("../artifacts/", "artifacts/")
    
    # Externalize links pointing above docs root to GitHub source repository
    github_blob = "https://github.com/xangma/diffgw/blob/main"
    markdown = markdown.replace("../src/", f"{github_blob}/src/")
    markdown = markdown.replace("../THIRD_PARTY_NOTICES.md", f"{github_blob}/THIRD_PARTY_NOTICES.md")
    markdown = markdown.replace("../CITATIONS.md", f"{github_blob}/CITATIONS.md")
    
    return markdown
