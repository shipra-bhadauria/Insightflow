import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# import all tools
from tools.describe       import describe_data
from tools.aggregate      import aggregate
from tools.trend          import trend_over_time
from tools.anomaly        import detect_anomaly
from tools.correlate      import correlate
from tools.chart          import make_chart
from tools.forecast       import forecast
from tools.what_if        import what_if


# MCP tool registry — each tool has name, description, and parameter schema
MCP_TOOLS = [
    {
        "name": "describe_data",
        "description": "Get summary statistics for all columns in the dataset. Use when the user wants an overview of the data.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "aggregate",
        "description": "Group data by a column and calculate mean, sum, count, min, or max of a value column.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_by":  {"type": "string", "description": "Column to group by"},
                "value_col": {"type": "string", "description": "Column to aggregate"},
                "agg":       {"type": "string", "enum": ["mean", "sum", "count", "min", "max"]},
                "dropna":    {"type": "boolean", "default": True},
            },
            "required": ["group_by", "value_col", "agg"],
        },
    },
    {
        "name": "trend_over_time",
        "description": "Calculate percentage change of a value column over time periods.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":  {"type": "string", "description": "Date column name"},
                "value_col": {"type": "string", "description": "Value column to trend"},
                "freq":      {"type": "string", "enum": ["ME", "W", "QE", "YE"], "default": "ME"},
            },
            "required": ["date_col", "value_col"],
        },
    },
    {
        "name": "detect_anomaly",
        "description": "Detect outliers in a numeric column using IQR method.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "Numeric column to check for outliers"},
            },
            "required": ["col"],
        },
    },
    {
        "name": "correlate",
        "description": "Calculate Pearson correlation between two numeric columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "col_a": {"type": "string", "description": "First numeric column"},
                "col_b": {"type": "string", "description": "Second numeric column"},
            },
            "required": ["col_a", "col_b"],
        },
    },
    {
        "name": "make_chart",
        "description": "Generate a chart and save as PNG. Use after aggregate or trend for visualisation.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["bar", "line", "scatter", "histogram"]},
                "x":    {"type": "string", "description": "X axis column"},
                "y":    {"type": "string", "description": "Y axis column"},
            },
            "required": ["kind", "x", "y"],
        },
    },
    {
        "name": "forecast",
        "description": "Forecast future values using Prophet time series model.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":  {"type": "string", "description": "Date column"},
                "value_col": {"type": "string", "description": "Value column to forecast"},
                "periods":   {"type": "integer", "default": 30},
                "freq":      {"type": "string", "enum": ["ME", "W", "QE", "YE"], "default": "ME"},
            },
            "required": ["date_col", "value_col"],
        },
    },
    {
        "name": "what_if",
        "description": "Simulate a percentage change on a column and show the impact.",
        "parameters": {
            "type": "object",
            "properties": {
                "col":        {"type": "string", "description": "Column to apply change to"},
                "change_pct": {"type": "number", "description": "Percentage change — positive or negative"},
                "group_by":   {"type": "string", "description": "Optional column to break down impact by"},
            },
            "required": ["col", "change_pct"],
        },
    },
]

# map tool names to actual functions
TOOL_FUNCTIONS = {
    "describe_data":   describe_data,
    "aggregate":       aggregate,
    "trend_over_time": trend_over_time,
    "detect_anomaly":  detect_anomaly,
    "correlate":       correlate,
    "make_chart":      make_chart,
    "forecast":        forecast,
    "what_if":         what_if,
}


def list_tools() -> list:
    return MCP_TOOLS


def get_tool_schema(tool_name: str) -> dict:
    for tool in MCP_TOOLS:
        if tool["name"] == tool_name:
            return tool
    raise ValueError(f"Tool not found: {tool_name}")


def call_tool(
    tool_name: str,
    df: pd.DataFrame,
    tool_args: dict
) -> dict:
    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_fn = TOOL_FUNCTIONS[tool_name]

    if tool_name == "describe_data":
        return tool_fn(df)

    return tool_fn(df, **tool_args)


def handle_mcp_request(request: dict, df: pd.DataFrame = None) -> dict:
    method = request.get("method")

    # tools/list — return all available tools
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id":      request.get("id"),
            "result":  {"tools": list_tools()},
        }

    # tools/call — execute a specific tool
    elif method == "tools/call":
        params    = request.get("params", {})
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if df is None:
            return {
                "jsonrpc": "2.0",
                "id":      request.get("id"),
                "error":   {"code": -32600, "message": "No DataFrame provided"},
            }

        try:
            result = call_tool(tool_name, df, tool_args)
            return {
                "jsonrpc": "2.0",
                "id":      request.get("id"),
                "result":  result,
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id":      request.get("id"),
                "error":   {"code": -32603, "message": str(e)},
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id":      request.get("id"),
            "error":   {"code": -32601, "message": f"Method not found: {method}"},
        }


if __name__ == "__main__":
    print("=== MCP Server Test ===\n")

    # test 1 — list tools
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    response = handle_mcp_request(request)
    tools = response["result"]["tools"]
    print(f"Tools available: {len(tools)}")
    for tool in tools:
        print(f"  {tool['name']:20s} — {tool['description'][:60]}")

    # test 2 — call a tool
    print("\n--- Calling aggregate tool ---")
    df = pd.read_csv("data/sales.csv")

    request2 = {
        "jsonrpc": "2.0",
        "id":      2,
        "method":  "tools/call",
        "params":  {
            "name":      "aggregate",
            "arguments": {
                "group_by":  "Region",
                "value_col": "Total Revenue",
                "agg":       "mean",
            }
        }
    }
    response2 = handle_mcp_request(request2, df=df)
    result    = response2["result"]
    print(f"Rows used: {result['rows_used']}")
    print(f"Top region: {max(result['result'], key=result['result'].get)}")
    print(f"\nMCP response format:")
    print(f"  jsonrpc: {response2['jsonrpc']}")
    print(f"  id:      {response2['id']}")
    print(f"  result:  dict with {len(result)} keys")