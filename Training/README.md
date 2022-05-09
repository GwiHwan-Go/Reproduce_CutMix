# If you run this on jupyter notebook, read below.

## To properly handle pytorch in local environment

### 1. Set a virtual environment(If you are using Local environment)

```bash
!pip install pyenv virtualenv
```
And then,
```
vim ~/.zshrc
```
input below lines in the opened file.
##### Mac OS X
```
export PYENV_ROOT=/usr/local/var/pyenv
eval "$(pyenv init -)"
eval "$(pyenv init --path)"

```
##### Linux
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```
After added above, just run
```
source ~/.zshrc
```
install the python version you want to install on the pyenv
you can see available list, you can see list from   ```pyenv install --list```
if you input  ```pyenv versions``` there will be a * mark, it is indicating the version that you're using.

Finally, Let's create a virtual environment.
```
pyenv virtualenv <versions> <name_of_your_env>
ex) pyenv virtualenv 3.7.3 torch_cv
```
You can uninstall your virtual environment through
``` pyenv uninstall <name_of_your_env> ```

Let's set your working space as the virtual environment we just made.
Move into your working space and input below line.

```
pyenv local <name_of_your_env>
```

Finally,
we can see installed libraries in the virtual environment with

```
pip list
```
In addition to thi

### 2. Implement on jupyter notebook

you can run below lines to set pyenv as the kernel of your jupyter notebook
```
!source activate torch_cv
!pip install ipykernel
!python -m ipykernel install --user --name torch_cv --display-name "torch_cv"
```

### 3. install matched torch and cuda

Here is what I did.

Firtst, install the conda.
```
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
add "export PATH="/home/username/anaconda3/bin:$PATH" to ~/.bashrc
source ~/.bashrc
```
And then, install matched version of pytorch tools.
For me, I'm using the version of cuda 11.7
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

```
Finally, check out whether intended version has installed or not.
```
pip list
python -> import torch -> torch.__version__

```
This kind of problem really drove me crazy since I didn't know what is happening inside...