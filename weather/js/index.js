



/// dymanic element
/*
let input_ele=document.createElement('input');
input_ele.setAttribute('type','file');
input_ele.accept='.csv';
input_ele.hidden=true;
document.body.appendChild(input_ele);
*/

////////////////////////////
/*
let form_ele=document.createElement("form");
let submit_ele=document.createElement('input');
submit_ele.hidden=true;
form_ele.setAttribute('type','submit');
form_ele.appendChild(input_ele);
form_ele.appendChild(submit_ele);
document.body.appendChild(form_ele);
*/
////////////////////////////


function show_date(){
    let today =new Date();
    const [month, day, year,hour,min,sec]=[today.getMonth(),today.getDay(),today.getFullYear(),today.getHours(),today.getMinutes(),today.getSeconds()];
    //console.log(month+"月"+day+"號"+year+"年"+hour+"點"+min+"分"+sec+"秒");
}

setInterval(show_date,1000);


/*
function processFile() {
    var fileSize = 0;
    //get file
    var theFile = input_ele;
    console.log(theFile.file);
     var regex = /^([a-zA-Z0-9\s_\\.\-:])+(.csv|.txt)$/;
     //check if file is CSV
     if (regex.test(theFile.value.toLowerCase())) {
     //check if browser support FileReader
        if (typeof (FileReader) != "undefined") {
       //get table element
        var table = document.getElementById("myTable");
        var headerLine = "";
        //create html5 file reader object
        var myReader = new FileReader();
        // call filereader. onload function
        myReader.onload = function(e) {
            var content = myReader.result;
            //split csv file using "\n" for new line ( each row)
            var lines = content.split("\r");
            //loop all rows
            for (var count = 0; count < lines.length; count++) {
                //create a tr element
                var row = document.createElement("tr");
                //split each row content
                var rowContent = lines[count].split(",");
                //loop throw all columns of a row
                for (var i = 0; i < rowContent.length; i++) {
                   //create td element 
                    var cellElement = document.createElement("td");
                    if (count == 0) {
                        cellElement = document.createElement("th");
                    } else {
                        cellElement = document.createElement("td");
                    }
                    //add a row element as a node for table
                    var cellContent = document.createTextNode(rowContent[i]);
                    
                    cellElement.appendChild(cellContent);
                    //append row child
                    row.appendChild(cellElement);
                }
                //append table contents
                myTable.appendChild(row);
            }
        }
         //call file reader onload
          myReader.readAsText(theFile.files[0]);
        }
        else 
        {
              alert("This browser does not support HTML5.");
        }
        
    }
    else {
                alert("Please upload a valid CSV file.");
    }
    return false;
}

processFile();*/

/*
let file_ele=document.getElementById("myfile").files[0];

console.log(file_ele)



file_ele.onchange=function(){

    readFile(file_ele);
}

function readFile(input) {
    let file = input.files[0];
  
    let reader = new FileReader();
  
    reader.readAsText(file);
  
    reader.onload = function() {
      console.log(reader.result);
    };
  
    reader.onerror = function() {
      console.log(reader.error);
    };
  
  }*/