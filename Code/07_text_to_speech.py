# to speech conversion
from gtts import gTTS
from playsound import playsound  
  
# This module is imported so that we can 
# play the converted audio
import os
  
mytext = 'Testing Voice'
  
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("welcome.mp3")
playsound("welcome.mp3")