cd ../preprocess/
python extract_noun.py ny
cd ../run/

cd ../embedding/scripts/
./ny.sh
cd ../../run/

cd ../split/scripts/
./ny.sh
cd ../../run/

cd ../vmf/scripts/
./ny.sh
cd ../../run/

cd ../classify/scripts/
./ny.sh
cd ../../run/
