#!/usr/bin/env python3
"""
Install external C++ dependencies for pyRTX
This is automatically called during pip install
"""
import os
import sys
import subprocess
import urllib.request
import tarfile
import shutil
import json
import platform
import zipfile
import re
from pathlib import Path
from typing import Optional

# Configuration flag to enable/disable CGAL and Boost installation
INSTALL_CGAL_BOOST = False  # Set to True to enable in future releases

class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def log_info(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}", file=sys.stderr)

def log_step(msg: str):
    print(f"{Colors.BLUE}[STEP]{Colors.NC} {msg}")

def run_command(cmd: list, cwd: Optional[Path] = None, check=True) -> subprocess.CompletedProcess:
    """Run shell command with error handling"""
    log_info(f"Running: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            log_error(f"Error output: {e.stderr}")
        raise

def download_file(url: str, dest_path: Path):
    """Download file with progress indication"""
    if dest_path.exists():
        log_info(f"{dest_path.name} already exists, skipping download")
        return
    
    log_info(f"Downloading {url}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # New line after progress
    except Exception as e:
        log_error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def extract_tarball(filepath: Path, dest_dir: Path):
    """Extract tarball"""
    log_info(f"Extracting {filepath.name}")
    try:
        with tarfile.open(filepath) as tar:
            tar.extractall(dest_dir)
    except Exception as e:
        log_error(f"Failed to extract {filepath}: {e}")
        raise

def install_build_dependencies():
    """Install critical build dependencies from requirements-build.txt"""
    log_step("Installing build dependencies...")
    
    # Check if requirements-build.txt exists
    build_req_file = Path('requirements-build.txt')
    if not build_req_file.exists():
        log_warn("requirements-build.txt not found, installing minimal build dependencies")
        # Fallback to minimal dependencies
        build_deps = ['cython', 'numpy', 'setuptools', 'wheel']
        for dep in build_deps:
            try:
                log_info(f"Installing {dep}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                log_info(f"✓ {dep} installed successfully")
            except subprocess.CalledProcessError as e:
                log_error(f"✗ Failed to install {dep}: {e}")
                sys.exit(1)
    else:
        log_info(f"Installing from {build_req_file}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', str(build_req_file)
            ])
            log_info("✓ Build dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            log_error(f"✗ Failed to install build dependencies: {e}")
            sys.exit(1)

def install_main_dependencies():
    """Install main dependencies from requirements.txt"""
    log_step("Installing main dependencies...")
    
    # Check if requirements.txt exists
    main_req_file = Path('requirements.txt')
    if not main_req_file.exists():
        log_warn("requirements.txt not found, skipping main dependencies")
        return
    
    log_info(f"Installing from {main_req_file}")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '-r', str(main_req_file)
        ])
        log_info("✓ Main dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        log_warn(f"Some main dependencies failed to install: {e}")
        log_warn("Continuing with C++ dependency installation...")


class DependencyInstaller:
    def __init__(self, lib_dir: str = "./lib"):
        self.lib_dir = Path(lib_dir).resolve()
        self.lib_dir.mkdir(exist_ok=True)
        
        # Track installation state
        self.state_file = self.lib_dir / ".install_state.json"
        self.state = self._load_state()
        
        self.config = {
            'embree2': '2.17.7',
            'embree3': '3.13.5',
            'cgal': '5.6',
            'boost': '1.82.0'
        }
    
    def _load_state(self) -> dict:
        """Load installation state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        """Save installation state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _mark_complete(self, component: str):
        """Mark a component as installed"""
        self.state[component] = 'installed'
        self._save_state()
    
    def _is_installed(self, component: str) -> bool:
        """Check if component is already installed"""
        return self.state.get(component) == 'installed'
    
    def _get_platform_suffix(self):
        """Get platform-specific suffix for Embree packages"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'linux':
            return 'x86_64.linux', 'linux'
        elif system == 'darwin':  # macOS
            if machine == 'arm64':
                raise RuntimeError(
                    f"ARM64 macOS (Apple Silicon) is not yet supported.\n"
                    f"Embree does not provide pre-built binaries for arm64 macOS.\n"
                    f"Please use an x86_64 environment or build Embree from source."
                )
            else:
                return 'x86_64.macosx', 'darwin'
        else:
            raise RuntimeError(f"Unsupported platform: {system} on {machine}")
    
    def check_prerequisites(self):
        """Check if required tools are installed"""
        log_step("Checking prerequisites...")
        required_tools = ['git']
        
        missing = []
        for tool in required_tools:
            if not shutil.which(tool):
                missing.append(tool)
        
        if missing:
            log_error(f"Missing required tools: {', '.join(missing)}")
            log_error("Please install them before continuing")
            sys.exit(1)
        
        log_info("All prerequisites satisfied")
    
    def install_embree(self, version: str, component_name: str):
        """Install Embree library (platform-aware)"""
        if self._is_installed(component_name):
            log_info(f"{component_name} already installed, skipping")
            return
        
        log_step(f"Installing Embree {version}")
        
        # Get platform-specific information
        platform_suffix, platform_name = self._get_platform_suffix()
        
        # Construct URL and filename based on platform
        if platform_name == 'darwin':  # macOS
            if version.startswith('2'):
                archive_ext = 'tar.gz'
            else:
                archive_ext = 'zip'
        else:  # Linux
            archive_ext = 'tar.gz'
        
        url = f"https://github.com/embree/embree/releases/download/v{version}/embree-{version}.{platform_suffix}.{archive_ext}"
        filename = f"embree-{version}.{platform_suffix}.{archive_ext}"
        embree_dir_name = f"embree-{version}.{platform_suffix}"
        
        filepath = self.lib_dir / filename
        
        # Download
        download_file(url, filepath)
        
        # Extract based on archive type
        if archive_ext == 'zip':
            log_info(f"Extracting {filepath.name}")
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(self.lib_dir)
            except Exception as e:
                log_error(f"Failed to extract {filepath}: {e}")
                raise
        else:  # tar.gz
            extract_tarball(filepath, self.lib_dir)
        
        # Set up environment
        embree_dir = self.lib_dir / embree_dir_name
        env_script = embree_dir / "embree-vars.sh"
        
        if env_script.exists():
            log_info("Setting up Embree environment variables")
            with open(env_script) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('export'):
                        try:
                            var_assignment = line.split('export ')[1]
                            var, value = var_assignment.split('=', 1)
                            # Expand variables in the value
                            expanded_value = value.strip('"').strip("'")
                            # Replace ${VAR} and $VAR references
                            for match in re.finditer(r'\$\{?(\w+)\}?', expanded_value):
                                env_var = match.group(1)
                                if env_var in os.environ:
                                    expanded_value = expanded_value.replace(match.group(0), os.environ[env_var])
                            os.environ[var] = expanded_value
                            log_info(f"Set {var}={expanded_value[:50]}...")
                        except Exception as e:
                            log_warn(f"Could not parse line: {line}")
        else:
            log_warn(f"embree-vars.sh not found at {env_script}")
            log_warn("Setting up basic environment variables manually")
            # Set basic paths manually
            os.environ['EMBREE_ROOT_DIR'] = str(embree_dir)
            os.environ['EMBREE_INCLUDE_DIR'] = str(embree_dir / 'include')
            os.environ['EMBREE_LIB_DIR'] = str(embree_dir / 'lib')
            
            # Update PATH
            bin_dir = embree_dir / 'bin'
            if bin_dir.exists():
                os.environ['PATH'] = f"{bin_dir}:{os.environ.get('PATH', '')}"
            
            # Update library path (platform-specific)
            lib_dir = embree_dir / 'lib'
            if lib_dir.exists():
                if platform_name == 'darwin':
                    dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
                    os.environ['DYLD_LIBRARY_PATH'] = f"{lib_dir}:{dyld_path}" if dyld_path else str(lib_dir)
                else:
                    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                    os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{ld_path}" if ld_path else str(lib_dir)
        
        # Clean up archive
        filepath.unlink()
        
        self._mark_complete(component_name)
        log_info(f"Embree {version} installed successfully on {platform_name}")
        
    def install_python_embree(self):
        """Install python-embree bindings with correct RPATH/install_name"""
        component_name = 'python-embree'
        if self._is_installed(component_name):
            log_info(f"{component_name} already installed, skipping")
            return
        
        log_step("Installing python-embree")
        embree_dir = self.lib_dir / "python-embree"
        
        # Clone repository
        if not embree_dir.exists():
            log_info("Cloning python-embree repository")
            run_command(
                ['git', 'clone', 'https://github.com/sampotter/python-embree.git'],
                cwd=self.lib_dir
            )
        else:
            log_info("python-embree repository already cloned")
        
        # Apply compatibility fix
        log_info("Applying compatibility fix")
        embree_pyx = embree_dir / "embree.pyx"
        
        if embree_pyx.exists():
            with open(embree_pyx, 'r') as f:
                content = f.read()
            
            if 'rtcSetDeviceErrorFunction' in content and '# rtcSetDeviceErrorFunction' not in content:
                content = content.replace(
                    'rtcSetDeviceErrorFunction(self._device, simple_error_function, NULL);',
                    '# rtcSetDeviceErrorFunction(self._device, simple_error_function, NULL);  # Commented for compatibility'
                )
                
                with open(embree_pyx, 'w') as f:
                    f.write(content)
                log_info("Applied fix to embree.pyx")
            else:
                log_info("Fix already applied or not needed")
        
        # Get Embree paths (platform-aware)
        platform_suffix, platform_name = self._get_platform_suffix()
        
        embree3_dir = self.lib_dir / f"embree-{self.config['embree3']}.{platform_suffix}"
        embree_lib_dir = embree3_dir / 'lib'
        embree_include_dir = embree3_dir / 'include'
        embree_vars_script = embree3_dir / "embree-vars.sh"
        
        if not embree3_dir.exists():
            log_error(f"Embree directory not found: {embree3_dir}")
            log_error("Make sure Embree 3 is installed first")
            raise FileNotFoundError(f"Missing {embree3_dir}")
        
        log_info(f"Using Embree from: {embree3_dir}")
        log_info(f"Embree lib directory: {embree_lib_dir}")
        
        # Set up Embree environment
        if embree_vars_script.exists():
            log_info(f"Loading Embree environment from {embree_vars_script}")
            bash_command = f"source {embree_vars_script} && env"
            
            try:
                result = subprocess.run(
                    ['bash', '-c', bash_command],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                embree_env = {}
                for line in result.stdout.split('\n'):
                    if '=' in line:
                        key, _, value = line.partition('=')
                        embree_env[key] = value
            except subprocess.CalledProcessError as e:
                log_error(f"Failed to source embree-vars.sh: {e}")
                log_error(f"stderr: {e.stderr}")
                raise
        else:
            log_warn("embree-vars.sh not found, setting up environment manually")
            embree_env = os.environ.copy()
            embree_env['EMBREE_ROOT_DIR'] = str(embree3_dir)
            embree_env['EMBREE_INCLUDE_DIR'] = str(embree_include_dir)
            embree_env['EMBREE_LIB_DIR'] = str(embree_lib_dir)
        
        # Platform-specific linker flags
        if platform_name == 'darwin':
            # macOS uses install_name and rpath differently
            rpath_flag = f"-Wl,-rpath,{embree_lib_dir}"
            install_name_flag = f"-Wl,-install_name,@rpath/libembree3.dylib"
            ldflags = embree_env.get('LDFLAGS', '')
            embree_env['LDFLAGS'] = f"{rpath_flag} {install_name_flag} {ldflags}".strip()
            
            # Set DYLD_LIBRARY_PATH
            dyld_path = embree_env.get('DYLD_LIBRARY_PATH', '')
            embree_env['DYLD_LIBRARY_PATH'] = f"{embree_lib_dir}:{dyld_path}" if dyld_path else str(embree_lib_dir)
        else:  # Linux
            rpath_flag = f"-Wl,-rpath,{embree_lib_dir}"
            ldflags = embree_env.get('LDFLAGS', '')
            embree_env['LDFLAGS'] = f"{rpath_flag} {ldflags}".strip()
            
            # Set LD_LIBRARY_PATH
            ld_path = embree_env.get('LD_LIBRARY_PATH', '')
            embree_env['LD_LIBRARY_PATH'] = f"{embree_lib_dir}:{ld_path}" if ld_path else str(embree_lib_dir)
        
        # Show key Embree-related variables
        for key in ['EMBREE_ROOT_DIR', 'PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'CPATH', 'LDFLAGS']:
            if key in embree_env:
                value = embree_env[key]
                log_info(f"{key}={value[:80]}{'...' if len(value) > 80 else ''}")
        
        # Build and install
        log_info("Building python-embree with Embree environment and RPATH")
        
        try:
            result = subprocess.run(
                [sys.executable, 'setup.py', 'build_ext', '--inplace'],
                cwd=embree_dir,
                env=embree_env,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            log_info("Build successful")
        except subprocess.CalledProcessError as e:
            log_error(f"Build failed with exit code {e.returncode}")
            if e.stdout:
                log_error(f"Output: {e.stdout}")
            if e.stderr:
                log_error(f"Error: {e.stderr}")
            raise
        
        log_info("Installing python-embree")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '.'],
                cwd=embree_dir,
                env=embree_env,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            log_info("Installation successful")
        except subprocess.CalledProcessError as e:
            log_error(f"Installation failed with exit code {e.returncode}")
            if e.stdout:
                log_error(f"Output: {e.stdout}")
            if e.stderr:
                log_error(f"Error: {e.stderr}")
            raise
        
        # Verify RPATH/install_name
        self._verify_embree_rpath(embree_lib_dir, platform_name)
        
        self._mark_complete(component_name)
        log_info("python-embree installed successfully")

    def _verify_embree_rpath(self, expected_rpath, platform_name):
        """Verify that the installed embree module has the correct RPATH/install_name"""
        log_step("Verifying embree module RPATH/install_name")
        
        try:
            import site
            site_packages_dirs = site.getsitepackages()
            
            embree_so = None
            # macOS uses .so for Python extensions even though system libs use .dylib
            for sp_dir in site_packages_dirs:
                sp_path = Path(sp_dir)
                candidates = list(sp_path.glob('embree*.so'))
                if candidates:
                    embree_so = candidates[0]
                    break
            
            if not embree_so:
                log_warn("Could not find embree.so to verify")
                return
            
            log_info(f"Checking RPATH/install_name of {embree_so.name}")
            
            if platform_name == 'darwin':
                # Use otool on macOS
                if not shutil.which('otool'):
                    log_warn("otool not found, skipping verification")
                    return
                
                # Check dependencies
                result = subprocess.run(
                    ['otool', '-L', str(embree_so)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                log_info("Dependencies:")
                for line in result.stdout.split('\n')[1:]:  # Skip first line (filename)
                    if line.strip():
                        log_info(f"  {line.strip()}")
                
                # Check rpath
                result_rpath = subprocess.run(
                    ['otool', '-l', str(embree_so)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if 'LC_RPATH' in result_rpath.stdout:
                    log_info("✓ RPATH found in binary")
                    # Extract and show the actual rpath values
                    lines = result_rpath.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'LC_RPATH' in line:
                            # The path is usually 2 lines after LC_RPATH
                            if i + 2 < len(lines):
                                path_line = lines[i + 2].strip()
                                if 'path' in path_line:
                                    log_info(f"  {path_line}")
                else:
                    log_warn("⚠ No RPATH found in binary")
                    log_warn("You may need to set DYLD_LIBRARY_PATH manually")
                    
            else:  # Linux
                if not shutil.which('readelf'):
                    log_warn("readelf not found, skipping verification")
                    return
                
                result = subprocess.run(
                    ['readelf', '-d', str(embree_so)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                rpath_lines = [line for line in result.stdout.split('\n') 
                            if 'RPATH' in line or 'RUNPATH' in line]
                
                if rpath_lines:
                    log_info("Found RPATH/RUNPATH:")
                    for line in rpath_lines:
                        log_info(f"  {line.strip()}")
                    
                    rpath_str = ' '.join(rpath_lines)
                    if str(expected_rpath) in rpath_str:
                        log_info(f"✓ Embree library path is in RPATH")
                    else:
                        log_warn(f"⚠ Embree library path NOT found in RPATH")
                        log_warn(f"Expected: {expected_rpath}")
                else:
                    log_warn("No RPATH/RUNPATH found")
                    log_warn("You may need to set LD_LIBRARY_PATH manually")
            
        except Exception as e:
            log_warn(f"Could not verify RPATH/install_name: {e}")
        
    def install_cgal_boost(self):
        """Install CGAL and Boost (disabled for this release)"""
        if not INSTALL_CGAL_BOOST:
            log_warn("=" * 60)
            log_warn("CGAL and Boost installation is disabled in this release")
            log_warn("Set INSTALL_CGAL_BOOST = True to enable in future")
            log_warn("=" * 60)
            return
        
        if self._is_installed('cgal') and self._is_installed('boost'):
            log_info("CGAL and Boost already installed, skipping")
            return
        
        log_step("Installing CGAL and Boost")
        
        # Install CGAL
        if not self._is_installed('cgal'):
            cgal_version = self.config['cgal']
            cgal_url = f"https://github.com/CGAL/cgal/releases/download/v{cgal_version}/CGAL-{cgal_version}.tar.xz"
            cgal_file = self.lib_dir / f"CGAL-{cgal_version}.tar.xz"
            
            download_file(cgal_url, cgal_file)
            extract_tarball(cgal_file, self.lib_dir)
            cgal_file.unlink()
            
            self._mark_complete('cgal')
            log_info("CGAL installed successfully")
        
        # Install Boost
        if not self._is_installed('boost'):
            boost_version = self.config['boost']
            boost_ver_underscore = boost_version.replace('.', '_')
            boost_url = f"https://archives.boost.io/release/{boost_version}/source/boost_{boost_ver_underscore}.tar.gz"
            boost_file = self.lib_dir / f"boost_{boost_ver_underscore}.tar.gz"
            
            download_file(boost_url, boost_file)
            extract_tarball(boost_file, self.lib_dir)
            boost_file.unlink()
            
            self._mark_complete('boost')
            log_info("Boost installed successfully")
    
    def install_aabb_binder(self):
        """Install AABB binder (disabled when CGAL/Boost are disabled)"""
        if not INSTALL_CGAL_BOOST:
            log_warn("=" * 60)
            log_warn("AABB binder installation is disabled (requires CGAL/Boost)")
            log_warn("Set INSTALL_CGAL_BOOST = True to enable in future")
            log_warn("=" * 60)
            return
        
        component_name = 'aabb-binder'
        if self._is_installed(component_name):
            log_info(f"{component_name} already installed, skipping")
            return
        
        log_step("Installing AABB binder")
        aabb_dir = self.lib_dir / "py-cgal-aabb"
        
        # Clone repository
        if not aabb_dir.exists():
            log_info("Cloning py-cgal-aabb repository")
            run_command(
                ['git', 'clone', 'https://github.com/steo85it/py-cgal-aabb.git'],
                cwd=self.lib_dir
            )
        else:
            log_info("py-cgal-aabb repository already cloned")
        
        # Setup paths
        boost_ver = self.config['boost'].replace('.', '_')
        boost_path = (self.lib_dir / f"boost_{boost_ver}").resolve()
        cgal_path = (self.lib_dir / f"CGAL-{self.config['cgal']}" / "include").resolve()
        
        log_info(f"Boost path: {boost_path}")
        log_info(f"CGAL path: {cgal_path}")
        
        # Modify setup.py to add include directories
        setup_py = aabb_dir / "setup.py"
        
        with open(setup_py, 'r') as f:
            content = f.read()
        
        # Check if already modified
        if str(boost_path) not in content:
            log_info("Modifying setup.py to add include paths")
            lines = content.split('\n')
            
            # Find the line with Extension and add include_dirs
            for i, line in enumerate(lines):
                if 'Extension(' in line:
                    # Find the closing parenthesis
                    j = i
                    while ')' not in lines[j] or lines[j].count('(') > lines[j].count(')'):
                        j += 1
                    
                    # Insert include_dirs before the closing parenthesis
                    indent = ' ' * (len(lines[j]) - len(lines[j].lstrip()))
                    include_line = f'{indent}include_dirs=["{boost_path}", "{cgal_path}"],'
                    lines.insert(j, include_line)
                    break
            
            with open(setup_py, 'w') as f:
                f.write('\n'.join(lines))
            
            log_info("setup.py modified successfully")
        else:
            log_info("setup.py already contains include paths")
        
        # Build and install
        log_info("Building AABB binder")
        run_command([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=aabb_dir)
        
        log_info("Installing AABB binder")
        run_command([sys.executable, '-m', 'pip', 'install', '.'], cwd=aabb_dir)
        
        self._mark_complete(component_name)
        log_info("AABB binder installed successfully")
    
    def cleanup_downloads(self):
        """Remove downloaded archives to save space"""
        log_step("Cleaning up downloaded files")
        for pattern in ['*.tar.gz', '*.tar.xz', '*.zip']:
            for filepath in self.lib_dir.glob(pattern):
                log_info(f"Removing {filepath.name}")
                filepath.unlink()
    
    def install_all(self):
        """Run complete installation"""
        try:
            log_info("=" * 60)
            log_info("pyRTX Dependencies Installation")
            log_info("=" * 60)
            
            # Step 1: Install Python build dependencies first
            install_build_dependencies()
            
            # Step 2: Install main Python dependencies
            install_main_dependencies()
            
            # Step 3: Check system prerequisites
            self.check_prerequisites()
            
            # Step 4: Install C++ dependencies
            log_info("=" * 60)
            log_info("Installing C++ Dependencies")
            log_info("=" * 60)
            
            self.install_embree(self.config['embree2'], 'embree2')
            self.install_embree(self.config['embree3'], 'embree3')
            self.install_python_embree()
            
            # CGAL/Boost/AABB installation (disabled for this release)
            self.install_cgal_boost()
            self.install_aabb_binder()
            
            self.cleanup_downloads()
            
            log_info("=" * 60)
            log_info("Installation completed successfully!")
            if not INSTALL_CGAL_BOOST:
                log_warn("Note: CGAL, Boost, and AABB binder were skipped")
            log_info("=" * 60)
            
        except Exception as e:
            log_error(f"Installation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    installer = DependencyInstaller()
    installer.install_all()
