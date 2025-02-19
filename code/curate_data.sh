docker run -it -p 8080:8080 -p 8088:8088 --rm --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:24.12

# uninstall nemo-curator installed by the container and install the latest version instead
pip uninstall nemo-curator -y
rm -r /opt/NeMo-Curator
git clone https://github.com/NVIDIA/NeMo-Curator.git /opt/NeMo-Curator
python -m pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com /opt/NeMo-Curator[all]

python -m pip install --upgrade pip

# install dependency for running data curation
pip install -r requirements.txt

# might have issue with opencv version 
pip install qgrid
pip uninstall --yes $(pip list --format=freeze | grep opencv)
# might not need this
#rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
pip install opencv-python-headless

# run data curation pipeline
python3 data_curation.py