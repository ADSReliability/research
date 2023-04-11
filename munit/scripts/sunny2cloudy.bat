call "D:\ANACONDA\Scripts\activate.bat" D:\ANACONDA
call conda activate pytorch
python ../test.py --config ../configs/sunny2cloudy.yaml --input ../inputs/sunny2cloudy.jpg --output_folder ../outputs --checkpoint ../models/sunny2cloudy.pt --a2b 1 --num_style 1
pause