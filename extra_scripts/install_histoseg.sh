echo "Creating Environment using conda"
conda create -n histoseg python=3.6 anaconda -y
conda init bash
echo "Activating the current environment"
conda activate histoseg
echo "Pulling latest torch torchvision with cudatoolkit=10.2"
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y
echo "Installing openslide"
conda install -c bioconda openslide -y
echo "Installing libvips and pyvips"
conda install -c conda-forge libvips pyvips -y
echo "Installing Pytorch Lightning"
pip install pytorch-lightning -y

echo "Printing all the installe modules"
conda list
