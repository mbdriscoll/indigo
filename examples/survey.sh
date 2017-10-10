

echo $1

for OLEVEL in -O0 -O1 -O2 -O3 -O4; do
   echo $OLEVEL
   python examples/pics.py ../data/uwute1_highres/scan.h5 --backend $1 --debug 10 -i 10 $OLEVEL --crop TIME:1,COIL:4 2>&1 | grep "event='iter'" | grep "it=8" | cut -d= -f2 | cut -d, -f1
done
