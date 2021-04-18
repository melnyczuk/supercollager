# Supercollager
A service for cutting up images and putting them together.

## Building the executable
Supercollager can be built into a standalone executable using
```
pipenv run build
```

Ensure the shared cython binaries are used by including the `PYTHON_CONFIGURE_OPTS='--enable-framework'` flag when creating the venv.

This is captured in the `rebuild-venv` script:
```
pipenv run rebuild-venv
``` 

