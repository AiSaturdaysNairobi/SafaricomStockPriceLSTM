# SafaricomStockPriceLSTM
Use an LSTM to fit time series for Safaricom stock price.

## Getting the data

The first step is to obtain some data. Safaricom stock prices are available at [https://www.investing.com/equities/safaricom-historical-data
](https://www.investing.com/equities/safaricom-historical-data
).

There are several ways to do this. In a Linux terminal, the following command:

```bash
wget https://www.investing.com/equities/safaricom-historical-data -o safaricom-historical-data.html
```

Gets the file from the web and saves it as safaricom-historical-data.html

The resulting file has html markup, yet all that is desired are stock prices. These need to be extracted. Parsing the
html file can be done in a number of languages. [Python](https://www.python.org/) and [Beautiful soup](https://www.crummy.com/software/BeautifulSoup/) are often used. However, the html file
has a structure that can be exploited so that a simple [C](http://www.open-std.org/jtc1/sc22/wg14/) program can be used. The program uses the html file _safaricom-historical-data.html_ and produces two files with the parsed tables,
_safaricom-historical-data2.txt_ and _safaricom-historical-data.txt_ 

```C
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define MAXCHAR 10000

int main()
{
        FILE *fpr;
        FILE *fpw;
        FILE *fpw2;
        char str[MAXCHAR];
        char* filenamer="safaricom-historical-data.html";
        char* filenamew="safaricom-historical-data.txt";
        char* filenamew2="safaricom-historical-data2.txt";
        int i = 0;
        int j = 0;
        int stread = 0;
        char *ret;
        char st4[]="results_box";
        char st5[]="<tr>";
        char st6[]="</tr>";
        char st7[]="</tbody>";
        char writestr[MAXCHAR];
        int count = 0;
        fpr  = fopen(filenamer, "r");
        fpw  = fopen(filenamew, "w");
        fpw2 = fopen(filenamew2, "w"); 
        while(fgets(str, MAXCHAR, fpr) != NULL)
        {
                ret = strstr(str,st4);
                if(ret!=NULL)
                {
                        stread=1;
                        i=0;
                }
                if((stread==1)&&(i>12))
                {
                        ret = strstr(str,st6);
                        if(ret == NULL)
                        {
                                ret = strstr(str,st5);
                                if(ret == NULL)
                                {
                                        count++;
                                        j=0;
                                        while (str[j]!='>') j++;
                                        j++;
                                        while (str[j]!='<')
                                        {
                                                fprintf(fpw,"%c",str[j]);
                                                if(count==3) fprintf(fpw2,"%c",str[j]);
                                                j++;
                                        }
                                        if(count==6)
                                        {
                                                fprintf(fpw,"\n");
                                                fprintf(fpw2,"\n");
                                                count=0;
                                        }else{
                                                fprintf(fpw,";");
                                        }
                                }
                        }
                }
                i++;
                ret = strstr(str,st7);
                if(ret!=NULL)
                {
                        stread=0;
                }

        }
        fclose(fpr);
        fclose(fpw);
        fclose(fpw2);

        return 0;
}

```

Compile the C program and then run it in the same directory as the downloaded html page. Suppose the program is named
parse.c, then the following sequence of commands should produce the data in a format that is useable as input to Nico Jimenez's Python LSTM [code](https://github.com/nicodjimenez/lstm/blob/master/lstm.py):

```
gcc parse.c -o parse
./parse
```
You should then have the files _safaricom-historical-data2.txt_ and _safaricom-historical-data.txt_ 

## Fitting the data

Place the file [lstm.py](https://github.com/nicodjimenez/lstm/blob/master/lstm.py) in the same directory as the data. In addition, place the Python driver program in the file below in the same directory:

```Python
import numpy as np
import matplotlib.pyplot as plt
from lstm import LstmParam, LstmNetwork


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    max_x = 35.0
    y_list = []
    with open('safaricom-historical-data2.txt') as f:
        for line in f:
            data= line.split()
            y_list.append(float(data[0])/max_x)
    print(y_list)

    #y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(10000):
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])

        print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.apply_diff(lr=0.001)
        lstm_net.x_list_clear()
    y_pred = []
    for ind in range(len(y_list)):
        y_pred.append(lstm_net.lstm_node_list[ind].state.h[0])
    print(y_pred)
    plt.plot(y_pred,'r:',label='predicted')
    plt.plot(y_list,'b--',label='actual')
    plt.xlabel("day")
    plt.ylabel("price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example_0()
```

As a driver to predict the stock price based on historical data. Example program output is in the figure below:

![](images/Figure_1.png)

The fit is reasonable, though not great. Since the model uses only historical time series data, and does not really incorporate economic reasoning for changes in stock price, it should be used with great caution.

## References

* Nico Jimenez, _Simple LSTM_, [blog](http://nicodjimenez.github.io/2014/08/08/lstm.html), [code](https://github.com/nicodjimenez/lstm)
* Alexander Xavier, _Predicting stock prices with lstm_, [medium article](https://medium.com/neuronio/predicting-stock-prices-with-lstm-349f5a0974d4) [code](https://github.com/alexavierc/LSTM-Stock-Prices)
* Lilian Wengweng, _Predict stock prices using RNN_, [blog post](https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html)
* _RNN w/ LSTM cell example in TensorFlow and Python_, [tutorial](https://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/)
* Sebastian Otte, Dirk Krechel and Marcus Liwicki, _JANNlab_, [code repository](https://github.com/jannlab/jannlab)
* Andrej Karpathy, _The unreasonable effectiveness of recurrent neural networks_, [blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)



