Here is the protocol of the free and fixed parameters:

```
index     free parameters               fixed parameters     

41        Mabs, L_bar, b, H_0, Omega_m         -            #Should be 51!

31        L_bar, b, H_0, Omega_m            Mabs
32        Mabs, L_bar, H_0                  b
33        Mabs, L_bar, b                    H_0
34        Mabs, b, H_0, Omega_m             L_bar           #Should be 44!

21        L_bar, b                          Mabs, H_0
22        L_bar, H_0                        Mabs, b
23        Mabs, L_bar                       b, H_0

1         L_bar                             Mabs, b, H_0
```
