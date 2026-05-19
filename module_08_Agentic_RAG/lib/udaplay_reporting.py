#Reporting utilities 

from __future__ import annotations

import json
from typing import Any, Dict, List


def format_text_report(report: Dict[str, Any]) -> str:
    lines = [
        "Answer:",
        str(report.get("answer", "")),
        "",
        f"Confidence: {report.get('confidence', 'unknown')}",
        f"Source Used: {report.get('source_used', 'unknown')}",
        f"Fallback Used: {report.get('fallback_used', False)}",
        "",
        "Tools Used:",
    ]
    for tool_name in report.get("tools_used", []):
        lines.append(f"- {tool_name}")

    lines.append("")
    lines.append("Sources:")
    sources: List[Dict[str, Any]] = report.get("sources", [])
    if not sources:
        lines.append("- No sources available")
    else:
        for source in sources:
            if source.get("url"):
                lines.append(f"- {source.get('title', 'Untitled')} ({source.get('url')})")
            else:
                lines.append(f"- {source.get('title', 'Untitled')} [{source.get('type', 'local')}]")
    return "\n".join(lines)


def format_json_report(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
