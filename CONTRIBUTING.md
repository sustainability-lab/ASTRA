## Contributing to this repo
- Checkout the following video series if you are new to the GitHub ecosystem:
    - [Git and GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk)
    - [Complete Git and GitHub Tutorial for Beginners in हिंदी](https://www.youtube.com/watch?v=Ez8F0nW6S-w)
    - [Git and GitHub Tutorial for Beginners](https://youtu.be/tRZGeaHPoaw?si=H8apX-aWJKZQFlPi)
- **Don't clone this repo directly**. Instead, fork it and then clone your forked repo, create a new branch so that you can push your changes to that branch in your forked repo. Keep forked repo's `main` branch up to date with this repo's `main` branch.
- Install the cloned repo in editable mode by running at the root of the repo:

```bash
pip install -e .
```

Editable mode allows you to make changes to the code and use the changes without having to reinstall the package.
- Add relevant tests for your changes in the `tests` folder. Use `pytest` to run the tests. If the tests pass successfully, you can push your changes to your forked repo and create a pull request to this repo.