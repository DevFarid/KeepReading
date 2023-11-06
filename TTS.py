from gtts import gTTS 
import sys
import pygame
import time

class KeepTalking():
    def __init__(self, lang='en', slowTalk=False) -> None:
        """Constructs a KeepTalking object that can talk.
        This class is wrapped around the gTTS & pygame library.
        With few additions, such as saving continous audio files for various processes.
        It also is a player library that allows one to play pre-record mp3 files.
        Create by Farid Kamizi on 10/30/2023.

        Args:
            lang (str, optional): The language for this object to operate on. Defaults to 'en'.
            slowTalk (bool, optional): I think this slows down the AIs voice. Defaults to False.
        """
        self.gtts = None
        self.lang = lang
        self.slow = slowTalk

    def save(self, text, location="tts/latest.mp3"):
        """Saves the text as an mp3 file.

        Args:
            text (str): The text to be saved in the language specified in the constructor.
        """
        self.gtts = gTTS(text=text, lang=self.lang, slow=self.slow)
        self.gtts.save(location)

    def play(self, file, vol=1.0):
        """Plays the file specified.

        Args:
            file (str): The file to be played.
        """
        pygame.init()
        pygame.mixer.music.load(file)
        pygame.mixer.music.set_volume(vol)
        pygame.mixer.music.play()

        # Keep the program running while the music is playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.quit()

    def talk(self, text, vol=1.0):
        """Immediately talks the text as soon as possible.

        Args:
            text (str): The text to be spoken in the language specified in the constructor.
        """
        self.save(text)
        self.play("tts/latest.mp3", vol)

    def saveMode(self):
        while(True):
            text = input(f"Enter a `{self.lang}` sentence to save: ")
            fileName = input(f"Enter a the name of this file to save as: ")
            if text == "\n" or fileName == "\n":
                break
            self.save(text, f"tts/pre-saved/{fileName}.mp3")
            
    def persistentAlertMode(self):
        pygame.init()
        pygame.mixer.music.load("tts/pre-saved/attentionRequired.mp3")
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play(loops=-1)

        # TODO: Might need meddle with threads so that the UI can function?
        # TODO: Also need a UI handling to stop instead of keyboard interrupt
        try:
            while True:
                # Keep the program running to allow the music to keep playing
                pass
        except KeyboardInterrupt:
            # Stop playing when ctrl+c is pressed
            pygame.mixer.music.stop()
            print("\nPlayback stopped")
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    kt = KeepTalking()
    kt.talk("Hello, my name is Farid")
    kt.persistentAlertMode()