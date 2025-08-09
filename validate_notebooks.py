#!/usr/bin/env python
"""
Validation script for Jupyter notebooks in PhysRevLett.133.106001 repository.
This script verifies that all notebooks execute successfully and dependencies are available.
"""

import sys
import os
import importlib
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Verify all required packages are installed with correct versions."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scipy', 'xarray',
        'qcodes', 'lmfit', 'holoviews', 'bokeh', 'jupyter', 'nbformat'
    ]
    
    print("Dependency Check Results:")
    print("=" * 40)
    missing_packages = []
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package}: {version}")
        except ImportError as e:
            print(f"❌ {package}: MISSING - {e}")
            missing_packages.append(package)
    
    print(f"\nPython Version: {sys.version}")
    return len(missing_packages) == 0

def check_data_files():
    """Verify all required data files are present."""
    required_files = {
        'D-SQUID-06.db': 146_000_000,  # ~146 MB
        'Extract_Critical_Current.ipynb': 1_000_000,  # ~1 MB
        'Fits_and_Figures.ipynb': 3_000_000,  # ~3 MB
    }
    
    print("\nData File Check:")
    print("=" * 40)
    
    all_present = True
    for filename, min_size in required_files.items():
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            size_mb = file_size / (1024**2)
            if file_size >= min_size:
                print(f"✅ {filename}: {size_mb:.1f} MB")
            else:
                print(f"⚠️  {filename}: {size_mb:.1f} MB (smaller than expected)")
        else:
            print(f"❌ {filename}: MISSING")
            all_present = False
    
    # Check for CSV files
    csv_files = [f for f in os.listdir('.') if f.startswith('CPR_id') and f.endswith('.csv')]
    print(f"✅ Found {len(csv_files)} CSV files (CPR_id*.csv)")
    
    return all_present

def test_notebook_execution(notebook_path, timeout=300):
    """Test if a notebook executes successfully."""
    print(f"\nTesting {notebook_path}...")
    print("-" * 40)
    
    start_time = time.time()
    try:
        # Use jupyter nbconvert to test execution
        result = subprocess.run([
            'jupyter', 'nbconvert', '--execute', '--to', 'notebook',
            '--stdout', notebook_path
        ], capture_output=True, text=True, timeout=timeout)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {notebook_path}: SUCCESS ({execution_time:.1f}s)")
            return True
        else:
            print(f"❌ {notebook_path}: FAILED")
            print("STDERR:", result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"⚠️  {notebook_path}: TIMEOUT after {execution_time:.1f}s")
        return False
    except Exception as e:
        print(f"❌ {notebook_path}: ERROR - {e}")
        return False

def main():
    """Main validation function."""
    print("PhysRevLett.133.106001 Notebook Validation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('README.md') or not os.path.exists('D-SQUID-06.db'):
        print("❌ Please run this script from the repository root directory")
        return False
    
    # Step 1: Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Missing dependencies. Please install required packages first.")
        return False
    
    # Step 2: Check data files
    data_ok = check_data_files()
    if not data_ok:
        print("\n❌ Missing required data files.")
        return False
    
    # Step 3: Test notebook execution
    notebooks = ['Extract_Critical_Current.ipynb', 'Fits_and_Figures.ipynb']
    execution_results = []
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            # Use different timeouts based on notebook complexity
            timeout = 120 if 'Extract' in notebook else 300
            success = test_notebook_execution(notebook, timeout)
            execution_results.append(success)
        else:
            print(f"❌ {notebook}: FILE NOT FOUND")
            execution_results.append(False)
    
    # Summary
    print("\nValidation Summary:")
    print("=" * 40)
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Data Files: {'✅ PASS' if data_ok else '❌ FAIL'}")
    
    for i, notebook in enumerate(notebooks):
        status = '✅ PASS' if execution_results[i] else '❌ FAIL'
        print(f"{notebook}: {status}")
    
    overall_success = deps_ok and data_ok and all(execution_results)
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED - Notebooks are ready for use!")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED - Please check the issues above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)