function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
check libcuda
check libcudart

function lib_installed2() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
function check2() { lib_installed2 $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
check2 libcudnn