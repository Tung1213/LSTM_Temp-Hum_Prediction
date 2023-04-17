

temp_array=[26,26,26,26,26,26,26]
humidity_array=[77,77,76,77,77,77,76]


temp=26
hum=77
##華氏(c)轉攝氏(F)
temp_f=temp*9/5+32
#####

###舒適度指數分類######


###總共有9個等級 4(86~88) 3(80~85) 2(76~79) 1(71~75) 0 (59~70) -1 -2 -3 -4
### 目前以 0~4 等級做分類

DI=temp_f-0.55*(1-0.01*hum)*(temp_f-58)

if DI<88 and DI>86:
    print(4)

elif DI<85 and DI>80:
    print(3)

elif DI<89 and DI>76:
    print(2)
    
elif DI<75 and DI>71:
    print(1)

else:
    print(0) 