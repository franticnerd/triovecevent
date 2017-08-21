cd ../preprocess/
python extract_noun.py toy
cd ../run/

cd ../embedding/scripts/
./toy.sh
cd ../../run/

cd ../split/scripts/
./toy.sh
cd ../../run/

cd ../vmf/scripts/
./toy.sh
cd ../../run/

cd ../classify/scripts/
./toy.sh
cd ../../run/
