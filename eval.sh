uname -a

export DATA=../data/uwute1_highres.h5

for BACKEND in 'cuda' 'customgpu'; do
  echo $BACKEND
  for OLEVEL in 0 1 2 3 4; do
    echo -n $OLEVEL: ' ' 
    python examples/pics.py $DATA --crop TIME:1,COIL:4 -i 100 --backend $BACKEND -O$OLEVEL --debug 10 2>&1 | grep ktime | cut -d= -f 2 | tr '\n' ', '
    echo
  done
done
