#!/usr/bin/env python3
"""
Validation script for YOLO-pose format output from GenerateData.py

Checks:
1. All coordinates are in [0, 1] range
2. Format matches YOLO-pose specification
3. Bounding boxes are valid
4. All 16 keypoints are present
"""

import sys
from pathlib import Path


def validate_yolo_pose_file(filepath: Path) -> dict:
    """Validate a single YOLO-pose annotation file.
    
    Returns dict with validation results.
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'num_people': 0
    }
    
    if not filepath.exists():
        results['valid'] = False
        results['errors'].append(f"File not found: {filepath}")
        return results
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    results['num_people'] = len(lines)
    
    for line_num, line in enumerate(lines, 1):
        parts = line.strip().split()
        
        # Check format: 1 class + 4 bbox + 48 keypoint values (16*3)
        expected_parts = 1 + 4 + (16 * 3)  # 53 total
        if len(parts) != expected_parts:
            results['valid'] = False
            results['errors'].append(
                f"Line {line_num}: Expected {expected_parts} values, "
                f"got {len(parts)}"
            )
            continue
        
        try:
            # Parse values
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            keypoints = [float(x) for x in parts[5:]]
            
            # Validate class ID
            if class_id != 0:
                results['warnings'].append(
                    f"Line {line_num}: Class ID is {class_id}, "
                    f"expected 0 (person)"
                )
            
            # Validate bounding box (normalized coordinates)
            x_c, y_c, w, h = bbox
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1):
                results['valid'] = False
                results['errors'].append(
                    f"Line {line_num}: Bounding box center out of range "
                    f"({x_c:.4f}, {y_c:.4f})"
                )
            
            if not (0 <= w <= 1 and 0 <= h <= 1):
                results['valid'] = False
                results['errors'].append(
                    f"Line {line_num}: Bounding box size out of range "
                    f"({w:.4f}, {h:.4f})"
                )
            
            # Validate keypoints (16 keypoints, each with x, y, visibility)
            for i in range(16):
                kp_x = keypoints[i * 3]
                kp_y = keypoints[i * 3 + 1]
                kp_v = int(keypoints[i * 3 + 2])
                
                if not (0 <= kp_x <= 1 and 0 <= kp_y <= 1):
                    results['valid'] = False
                    results['errors'].append(
                        f"Line {line_num}, Keypoint {i+1}: "
                        f"Coordinates out of range ({kp_x:.4f}, {kp_y:.4f})"
                    )
                
                if kp_v not in [0, 1, 2]:
                    results['valid'] = False
                    results['errors'].append(
                        f"Line {line_num}, Keypoint {i+1}: "
                        f"Invalid visibility flag {kp_v} (expected 0, 1, or 2)"
                    )
        
        except ValueError as e:
            results['valid'] = False
            results['errors'].append(
                f"Line {line_num}: Parse error - {str(e)}"
            )
    
    return results


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_yolo_pose.py <annotation_file.txt>")
        print("Example: python validate_yolo_pose.py /path/to/Annot/0001.txt")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    print(f"Validating: {filepath}")
    print("-" * 60)
    
    results = validate_yolo_pose_file(filepath)
    
    print(f"People detected: {results['num_people']}")
    print(f"Valid: {results['valid']}")
    
    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    if results['warnings']:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for warning in results['warnings'][:10]:
            print(f"  - {warning}")
        if len(results['warnings']) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more warnings")
    
    if results['valid'] and not results['errors']:
        print("\n✅ Annotation file is valid!")
        return 0
    else:
        print("\n❌ Annotation file has errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())
