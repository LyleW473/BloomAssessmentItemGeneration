"""
Filesystem helpers for locating and ordering course-content files.
- `get_all_paths_in_dir`: recursively list every file path under a directory.
- `sort_contents_by_week_number`: order directory names by an embedded week number.
"""
import os
from typing import List


def get_all_paths_in_dir(root_dir: str) -> List[str]:
    """
    Recursively gets all file paths in a directory.

    Args:
        root_dir (str): The root directory to search.
    """
    all_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            all_paths.append(os.path.join(dirpath, filename))
    return all_paths


def sort_contents_by_week_number(week_contents: List[str]) -> List[str]:
    """
    Sorts contents by a week number extracted from directory names.
    - Tries to extract a number from each directory name; if not possible, assigns infinity.

    Args:
        week_contents (List[str]): List of directory names to sort.
    """
    mapping = {}
    for week_dir in week_contents:
        try:
            week_num = int(''.join(filter(str.isdigit, week_dir)))
        except Exception:
            week_num = float('inf')
        mapping[week_dir] = week_num
    sorted_weeks = sorted(week_contents, key=lambda x: mapping[x])
    return sorted_weeks
