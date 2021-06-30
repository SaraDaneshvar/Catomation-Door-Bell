## Catomation-Door-Bell

### Package version

Python packages and versions are listed in `requirements.txt`. Install them for Raspberry Pi with the following:
```
virtualenv env
source env/bin/activate
pip3 install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl`
pip3 install -r requirements.txt
```

### Raspberry Pi system config

```
apt install libzip-dev libjpeg-dev libpython3-dev libf77dcl7 libatlas-base-dev libhdf5-dev libopenjp2-7 libilmbase23 nginx
sudo usermod -aG video $USER
```
