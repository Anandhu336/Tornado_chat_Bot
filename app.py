# Importing all necessary classes and functions from the tkinter module for GUI creation.
from tkinter import *

# Importing the get_response function and bot_name variable from a chat module.
from chat import get_response, bot_name

# Defining color constants for the GUI.
BG_GRAY = "#ABB2B9"  # Light gray color for certain background elements.
BG_COLOR = "#17202A"  # Dark color for the main background.
TEXT_COLOR = "#EAECEE"  # Light color for the text.

# Defining font styles to be used in the GUI.
FONT = "Helvetica 14"  # Regular Helvetica font, size 14.
FONT_BOLD = "Helvetica 13 bold"  # Bold Helvetica font, size 13.

# Defining the ChatApplication class, which contains all the functionality of the chatbot's GUI.
class ChatApplication:
    def __init__(self):
        # Initializing the main window using tkinter's Tk() and setting it up.
        self.window = Tk()
        self._setup_main_window()

    # Method to run the main loop of the tkinter application, keeping the window open.
    def run(self):
        self.window.mainloop()

    # Method to set up the main window, adding all GUI components.
    def _setup_main_window(self):
        # Setting the window title, disabling resizing, and setting the size and background color.
        self.window.title("TORNADO CHATBOT")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # Creating a header label with the chatbot's title and placing it at the top.
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome to AI CHATBOT", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # Adding a tiny divider line below the header.
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # Creating a text widget for displaying the chat messages.
        self.text_widget = Text(self.window, width=30, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # (Optional) Scroll bar configuration is commented out here.

        # Adding a label at the bottom to hold the message entry box and send button.
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # Creating an entry box for the user to type messages.
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)  # Bind the Enter key to send the message.

        # Adding a send button that will send the message when clicked.
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    # Method that is called when the Enter key is pressed or the send button is clicked.
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()  # Get the message from the entry box.
        self._insert_message(msg, "You")  # Insert the user's message into the chat.

    # Method to insert a message from the user and generate a response from the bot.
    def _insert_message(self, msg, sender):
        if not msg:  # If the message is empty, do nothing.
            return

        self.msg_entry.delete(0, END)  # Clear the entry box after getting the message.
        msg1 = f"{sender}: {msg}\n\n"  # Format the message with the sender's name.
        self.text_widget.configure(state=NORMAL)  # Enable the text widget to insert the message.
        self.text_widget.insert(END, msg1)  # Insert the user's message into the text widget.
        self.text_widget.configure(state=DISABLED)  # Disable the text widget to prevent editing.

        # Generate the bot's response using get_response() and insert it into the chat.
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)  # Scroll the text widget to the end to show the latest message.

# Check if the script is being run directly, and if so, create and run an instance of ChatApplication.
if __name__ == "__main__":
    app = ChatApplication()
    app.run()
