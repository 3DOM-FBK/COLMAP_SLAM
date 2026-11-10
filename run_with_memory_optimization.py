#!/usr/bin/env python3
"""
Run main.py with optimized CUDA memory settings
"""

import os
import subprocess
import sys

def main():
    # Set CUDA memory configuration to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Also set other memory optimization flags
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async operations
    
    print("Running with optimized CUDA memory settings:")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    
    # Run the main script
    try:
        # Use the same Python executable and pass all arguments
        cmd = [sys.executable, 'main.py'] + sys.argv[1:]
        result = subprocess.run(cmd, env=os.environ)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running main.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
