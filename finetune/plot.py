# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:38:55 2016

@author: Assaf
"""


import matplotlib.pyplot as plt


def flat(history_list,key):
        return sum([history.history[key] for history in history_list],[])
    
def graph_history(history_list):
    #comibe histories
    #keys= history_list[0].history.keys()

    # red dash r--  'bs' Blue Square  'g^' Green Triangles
    x_epocs= range(0,len(history_list))
    
    plt.plot(x_epocs, flat(history_list,'loss'),'b-',label='loss')
    plt.plot(x_epocs, flat(history_list,'val_loss'),'r-',label='val_loss')
    max_val = max( max(flat(history_list,'loss')), max(flat(history_list,'val_loss')))
    plt.axis([0, len(x_epocs), 0, max_val ])

    plt.legend()
    plt.show() 
 
    plt.plot(x_epocs, flat(history_list,'acc'),'b^--',label='acc')
    plt.plot(x_epocs, flat(history_list,'val_acc'),'r^--',label='val_acc')
    plt.axis([0, len(x_epocs), 0, 1])
    plt.legend()
    plt.show() 

    
    

#combine it to the first one...
def graph_history_3(history_list):
   
    
    #comibe histories

    x_epocs= range(0,len(history_list))

    l1 = plt.plot(x_epocs, flat(history_list,'loss'),'b-',label='loss')
    plt.plot(x_epocs, flat(history_list,'val_loss'),'r-',label='val_loss')
             
    l2 =plt.plot(        x_epocs, flat(history_list,'prob_loss'),'bo--',label='c2')
    plt.plot(         x_epocs, flat(history_list,'val_prob_loss'),'ro--',label='c2')
             
    l3= plt.plot(         x_epocs, flat(history_list,'prob_aux0_loss'),'bs--',label='c0')
    plt.plot(         x_epocs, flat(history_list,'val_prob_aux0_loss'),'rs--',label='c0')
             
    l4=plt.plot(         x_epocs, flat(history_list,'prob_aux1_loss'),'b^--',label='c1')
    plt.plot(         x_epocs, flat(history_list,'val_prob_aux1_loss'),'r^--',label='c1')
            
    plt.axis([0, len(x_epocs), 0, 1])
    plt.legend()
    plt.show()   
    
    plt.plot(
             
             x_epocs, flat(history_list,'prob_acc'),'bo--',
             x_epocs, flat(history_list,'val_prob_acc'),'ro--',
             
             x_epocs, flat(history_list,'prob_aux0_acc'),'bs--',
             x_epocs, flat(history_list,'val_prob_aux0_acc'),'rs--',
             
             x_epocs, flat(history_list,'prob_aux1_acc'),'b^--',
             x_epocs, flat(history_list,'val_prob_aux1_acc'),'r^--',
            )
    
    
           
    plt.axis([0, len(x_epocs), 0.5, 1.0])
    plt.show()          
    

   
    # red dash r--  'bs' Blue Square  'g^' Green Triangles
   
print 'done'  