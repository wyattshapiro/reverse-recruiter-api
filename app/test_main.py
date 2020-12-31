from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_classify_is_recruiter_true():
    response = client.post(
        "/classify",
        json={
            "message": "This isn't any normal engineering role, this is an opportunity for you to get involved in an exceptionally well-funded start-up, working with some of the most talented engineers in the market.  Whats in it for you? Endless opportunities for growth and development."
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        "is_recruiter": True
    }


def test_classify_is_recruiter_false():
    response = client.post(
        "/classify",
        json={
            "message": "This challenge is about picking out your high-impact, high-priority tasks every day, and then diving into it, letting go of the urge to go to distractions, the urge to procrastinate, the uncertainty that comes with these meaningful tasks. It's about overcoming our age-old habits of procrastination, and diving into our meaningful work."
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        "is_recruiter": False
    }


def test_classify_no_message():
    response = client.post(
        "/classify"
    )
    assert response.status_code == 422


