## Usage
You can run the codes in the folder `python` directly from `Google Colab` (make sure your session uses a GPU) as follows
```python
# clone repository
!git clone https://github.com/BI-DS/ELE-3909.git

# move to right folder and install requirements
%cd ELE-3909/vade/python
!pip install -r requirements.txt

# NOTE: after you install the requirements, you get a warning message about restar session.
# just click on the blue botton "Restart session" and execute the next code cell as usual

# move to right folder again
%cd ELE-3909/vade/python

# train VADE
!python train_vade.py --epochs 350 --load_weights 
```
