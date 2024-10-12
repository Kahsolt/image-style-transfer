@ECHO OFF

PUSHD %~dp0

git clone https://github.com/gsurma/style_transfer
git clone https://github.com/andrewstito/Image-Style-Transfer
git clone https://github.com/Suvoo/Image-Style-Transfer-Using-CNNs
git clone https://github.com/EliShayGH/deep-learning-style-transfer
git clone https://github.com/ali-gtw/ImageStyleTransfer-CNN

POPD

ECHO Done!
ECHO.

PAUSE
