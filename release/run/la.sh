cd ../preprocess/
python extract_noun.py la
cd ../run/

cd ../embedding/scripts/
./la.sh
cd ../../run/

cd ../split/scripts/
./la.sh
cd ../../run/

cd ../vmf/scripts/
./la.sh
cd ../../run/

cd ../classify/scripts/
./la.sh
cd ../../run/
