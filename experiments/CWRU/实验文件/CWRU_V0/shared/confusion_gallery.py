"""Common display rules for 180-window exclusion confusion matrices."""
import numpy as np

LABELS=['N','IR','OR','B']
SUPPORT=np.array([20,60,40,60])
CONDITIONS=[('clean','Clean'),('white_-4','-4 dB'),('white_-2','-2 dB'),
            ('white_+0','0 dB'),('white_+2','+2 dB'),('white_+4','+4 dB')]


def panel(ax,count,title):
    fraction=count/count.sum(1,keepdims=True)
    image=ax.imshow(fraction*100,cmap='Blues',vmin=0,vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{fraction[i,j]*100:.1f}%',ha='center',va='center',
                    color='white' if fraction[i,j]>.55 else 'black',fontsize=10)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,
           yticklabels=[f'{label} ({n})' for label,n in zip(LABELS,SUPPORT)],
           xlabel='Predicted',ylabel='True',title=f'{title}; ACC {100*np.trace(count)/count.sum():.2f}%')
    ax.tick_params(labelsize=10)
    return image
