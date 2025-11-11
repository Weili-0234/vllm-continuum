#!/usr/bin/env python3
"""
Calculate average job duration from scheduler_timestamps.

Each job's duration is calculated from the first Request_arrival_time
to the last Request_departure_time.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any


def load_scheduler_timestamps(input_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load scheduler_timestamps from the input directory."""
    timestamp_file = Path(input_dir) / "scheduler_timestamps"

    if not timestamp_file.exists():
        raise FileNotFoundError(f"scheduler_timestamps not found in {input_dir}")

    with open(timestamp_file, 'r') as f:
        data = json.load(f)

    return data


def calculate_job_duration(job_history: List[Dict[str, Any]]) -> float:
    """
    Calculate duration for a single job.

    Duration = last Request_departure_time - first Request_arrival_time
    """
    arrival_times = []
    departure_times = []

    for event in job_history:
        if "Request_arrival_time" in event:
            arrival_times.append(event["Request_arrival_time"])
        elif "Request_departure_time" in event:
            departure_times.append(event["Request_departure_time"])

    if not arrival_times or not departure_times:
        return None

    first_arrival = min(arrival_times)
    last_departure = max(departure_times)

    duration = last_departure - first_arrival
    return duration


def calculate_average_duration(timestamps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Calculate average job duration across all jobs."""
    job_durations = {}
    all_arrival_times = []
    all_departure_times = []

    for job_id, history in timestamps.items():
        duration = calculate_job_duration(history)
        if duration is not None:
            job_durations[job_id] = duration

        # Collect all arrival and departure times across all jobs
        for event in history:
            if "Request_arrival_time" in event:
                all_arrival_times.append(event["Request_arrival_time"])
            elif "Request_departure_time" in event:
                all_departure_times.append(event["Request_departure_time"])

    if not job_durations:
        return {
            "num_jobs": 0,
            "average_duration": 0,
            "total_duration": 0,
            "min_duration": 0,
            "max_duration": 0,
            "median_duration": 0,
            "percentile_95": 0,
            "percentile_99": 0,
            "job_durations": {}
        }

    durations = sorted(job_durations.values())

    # Calculate total_duration as largest departure - smallest arrival
    total_duration = max(all_departure_times) - min(all_arrival_times) if all_arrival_times and all_departure_times else 0

    # Calculate median
    n = len(durations)
    if n % 2 == 0:
        median_duration = (durations[n // 2 - 1] + durations[n // 2]) / 2
    else:
        median_duration = durations[n // 2]

    # Calculate percentiles
    def percentile(data, p):
        """Calculate percentile p (0-100) from sorted data."""
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    percentile_95 = percentile(durations, 95)
    percentile_99 = percentile(durations, 99)

    return {
        "num_jobs": len(job_durations),
        "average_duration": sum(durations) / len(durations),
        "total_duration": total_duration,
        "min_duration": min(durations),
        "max_duration": max(durations),
        "median_duration": median_duration,
        "percentile_95": percentile_95,
        "percentile_99": percentile_99,
        "job_durations": job_durations
    }


def save_results(results: Dict[str, Any], output_dir: str):
    """Save results to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

    # Also print summary to console
    print("\n=== Job Duration Statistics ===")
    print(f"Number of jobs: {results['num_jobs']}")
    print(f"Average duration: {results['average_duration']:.2f} seconds")
    print(f"Total duration: {results['total_duration']:.2f} seconds")
    print(f"Min duration: {results['min_duration']:.2f} seconds")
    print(f"Max duration: {results['max_duration']:.2f} seconds")
    print(f"Median duration: {results['median_duration']:.2f} seconds")
    print(f"95th percentile duration: {results['percentile_95']:.2f} seconds")
    print(f"99th percentile duration: {results['percentile_99']:.2f} seconds")
    print("=" * 35)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average job duration from scheduler_timestamps"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./continuum_exp",
        help="Directory containing scheduler_timestamps file (default: ./continuum_exp)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./continuum_exp",
        help="Directory to save results (default: ./continuum_exp)"
    )

    args = parser.parse_args()

    print(f"Loading scheduler_timestamps from {args.input_dir}...")
    timestamps = load_scheduler_timestamps(args.input_dir)

    print(f"Calculating job durations for {len(timestamps)} jobs...")
    results = calculate_average_duration(timestamps)

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
