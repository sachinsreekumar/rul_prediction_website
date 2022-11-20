import sqlite3
import hashlib
import pandas as pd
import streamlit as st
from PIL import Image

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

conn = sqlite3.connect('users.db')
c = conn.cursor()

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

# st.title("Aircraft Maintenance Prediction")

menu = ["Home","Login","SignUp","Logout"]
choice = st.sidebar.selectbox("Menu",menu)

session=0

if choice == "Home":
    # st.subheader("Home")
    image = Image.open('login_fan.jpg')
    st.image(image, width=620)

elif choice == "Login":
    st.subheader("Login Section")
    session=1
    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password",type='password')
    login_btn = st.sidebar.button("Login")
    if login_btn or session:
        # if password == '12345':
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))
        if result:

            # st.success("Logged In as {}".format(username))
            st.success("Logged In as {}".format(username))

        else:
            st.warning("Incorrect Username/Password")

elif choice == "Logout":
    session = 0



elif choice == "SignUp":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password",type='password')

    if st.button("Signup"):
        add_userdata(new_user,make_hashes(new_password))
        st.success("You have successfully created a valid Account")
        st.info("Go to Login Menu to login")


