icpx -fsycl -fopenmp -Wall -Ofast -g .\src\main.cpp .\src\io_routines.cpp .\src\stegano_routines.cpp -o .\builds\stegano.exe && .\builds\stegano.exe .\imgs\image_skyscraper.png .\imgs\logo_out.png .\imgs\Out\myout.png