


let temp_data=document.getElementsByClassName('degree');
let hum_data=document.getElementsByClassName('hum');

let temp_data_array=[]
let hum_data_array=[]


for(var i=0;i<temp_data.length;i++)temp_data_array[i]=temp_data[i].outerText.replace("oC","")

for(var i=0;i<hum_data.length;i++)hum_data_array[i]=hum_data[i].outerText.replace("%","")
