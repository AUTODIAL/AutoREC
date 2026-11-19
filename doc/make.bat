@ECHO OFF
setlocal
pushd "%~dp0"

if "%SPHINXBUILD%" == "" set "SPHINXBUILD=python -m sphinx"
if "%SPHINXOPTS%" == "" set "SPHINXOPTS=-n -W --keep-going"
set "BUILDTARGET=%~1"
if "%BUILDTARGET%" == "" set "BUILDTARGET=help"

%SPHINXBUILD% -M %BUILDTARGET% . _build %SPHINXOPTS% %O%
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
