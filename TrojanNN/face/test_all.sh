for fname in $1/*
do
    echo $fname
    python test_one_image.py $fname 2>log
done
