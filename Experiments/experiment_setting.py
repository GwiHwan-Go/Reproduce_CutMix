ex1= {'base' : {'cutmix' : 1, 'augmentation' : 'randomrotation(20)', 'log' : 'logs.txt'}, 
'comparison' : {'cutmix' : 0, 'augmentation' : 'randomrotation(20)', 'log' : 'no_cut_mic_logs.txt'}} 
ex2= {'base' : {'cutmix' : 1, 'augmentation' : None, 'log' : 'ex2-1.txt'}, 
'comparison' : {'cutmix' : 0, 'augmentation' :None, 'log' : 'ex2-2.txt'}}
ex3= {'base' : {'cutmix' : 1, 'augmentation' : 'normalize', 'log' : 'ex3-1.txt'}, 
'comparison' : {'cutmix' : 0, 'augmentation' :'normalize', 'log' : 'ex3-2.txt'}}
ex4= {'base' : {'cutmix' : 1, 'augmentation' : ['normalize','colorjitter'], 'log' : 'ex4-1.txt'}, 
'comparison' : {'cutmix' : 0, 'augmentation' :['normalize','colorjitter'], 'log' : 'ex4-2.txt'}}