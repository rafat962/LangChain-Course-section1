from dotenv import load_dotenv
import os
load_dotenv()




def main():
    print("Hello from course-resourses!")
    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("GOOGLE_API_KEY"))


if __name__ == "__main__":
    main()
