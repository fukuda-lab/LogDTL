# Install environment
```code 

First way:

    conda create -n log python==3.7.5
    
    conda activate log

    conda install numpy pandas gensim nltk
    pip install log2seq
    pip install python-crfsuite


Second way: (create environment by my envs.yml file)
    
        conda log create -f envs.yml (the first line in that file is the name of the environment)
        conda activate log
        pip install log2seq
        pip install python-crfsuite

```


2. Useful command
```code 

1) Activate, check and deactivate environment
    conda activate ai       

    conda list          (or)
    conda env list 
    conda info --envs    

    conda deactivate        
    
2) Check package inside the environment 
    conda list -n log                (if log hasn't activated)
    conda list                      (if log already activated)
    
3) Export to .yml for other usage.  
    source activate log                  (access to environment)
    conda env export > envs.yml     

4) Delete environment 
    conda remove --name log --all     (or)
    conda env remove --name log   
    
    conda info --envs   (check if it is deleted or not)
```
