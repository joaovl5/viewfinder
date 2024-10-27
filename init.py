from flask import Flask, Response, render_template
from camera import Camera, Pool
from dotenv import load_dotenv
import threading
import db


def run_app(pool: Pool):
    app = Flask(
        __name__,
        static_url_path="",
        static_folder="static",
        template_folder="templates",
    )

    @app.route("/")
    @app.route("/index")
    def index():
        return render_template(
            "index.j2",
            title="viewfinder",
            cameras=[{"id": str(idx)} for idx, x in enumerate(pool.cameras)],
        )

    @app.route("/video/<id>")
    def video(id):

        return Response(
            pool.get_frame(int(id)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    app.run(debug=False)


def main():
    load_dotenv()
    db_instance = db.Database()
    db_instance.setup_test_cameras()
    cameras = db_instance.get_cameras()

    pool = Pool([cam["source"][0] for cam in cameras])
    server_process = threading.Thread(target=run_app, args=(pool,))
    server_process.start()
    pool.start()
    pool.join()
    server_process.join()


if __name__ == "__main__":
    main()
