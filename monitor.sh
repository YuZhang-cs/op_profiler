while true; do (echo "%CPU %MEM ARGS $(date)" && ps -e -o pcpu,pmem,args,user --sort=pcpu | cut -d" " -f1-7 | tail) >> ps.log; sleep 5; done
