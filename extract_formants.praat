form Create_formant_script
    sentence Sound_file_name ""    
endform

Read from file... 'Sound_file_name$'
To Formant (burg)... 0 5 5000 0.025 50
formant_output$ = ""

for i from 1 to 3
    f = Get mean... i 0 0 Hertz
    formant_output$ = formant_output$ + string$(f) + " "
endfor

writeFileLine: "formants.txt", formant_output$
