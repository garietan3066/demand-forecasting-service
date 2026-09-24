# Dependency management

This project uses Python 3.11.

## Dependency inputs

- runtime.in: libraries used by the planned prediction service.
- training.in: runtime libraries plus data acquisition and preparation.
- development.in: training libraries plus testing and development tools.

## Windows development lock

development-windows-py311.txt contains resolved versions and package hashes.

Generate it from the project root:

```powershell
.\venv\Scripts\python.exe -m piptools compile --generate-hashes --strip-extras --allow-unsafe --output-file=requirements/development-windows-py311.txt requirements/development.in
```

The --allow-unsafe option includes installer dependencies such as pip and
setuptools in the lock. It does not disable hash or TLS verification.

Install the lock:

```powershell
python -m pip install --require-hashes -r requirements/development-windows-py311.txt
```

Do not edit the generated lock manually.

After changing dependency inputs:
1. Regenerate the lock.
2. Verify installation in a clean environment.
3. Run linting, tests, pip check, and pip-audit.
4. Review and commit the input and lock changes together.

## Platform support

The lock is generated for Windows and Python 3.11.

Linux CI currently installs from the input files to check compatibility.
It is not yet a locked deployment environment.

A separately generated and tested Linux lock is required before deployment.

The root requirements.txt and requirements-dev.txt are legacy bootstrap
lists. Prefer the Windows lock for verified Windows development.