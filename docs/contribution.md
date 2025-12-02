# Repository Contribution & Security Rules

This document defines the **rules and conventions** that all developers and researchers **must follow** when contributing to this repository.

---

## Git Commit Rules

### **Commit Message Format**

Use the following structure for all commits:

```
fix: short summary or message
feat: short summary or message
docs: short summary or message
refactor: short summary or message
test: short summary or message
chore: short summary or message
```

**Example:**

```
fix: handle null value in vectors
feat: add user schema and update endpoint
docs: update api for new agents
```

---

## Git Branch Naming Convention

Use this format for branches:

```
fix/subject
feat/subject
docs/subject
refactor/subject
```

**Examples:**

```
fix/tokens
feat/feature-extractor
refactor/database-service
```

---

## Git History

* Always **keep a linear history**.
* Use **rebase** instead of merge when updating from `main`.
* Avoid merge commits unless explicitly required.

**Example workflow:**

```bash
git fetch origin
git rebase origin/main
```

---

## Security & Data Rules

### **Never commit sensitive data**

* **Do not** commit **database connection credentials**.

  * Only include **environment variable keys**, without actual values.
  * Example:

    ```env
    DATABASE_URL=
    ```
* **Do not** commit **API keys or secrets**.

  * Use placeholder keys only:

    ```ts
    const API_KEY = process.env.MY_API_KEY;
    ```
* **Do not** include **user data** or **private company information** in the repository.

  * This includes emails, names, analytics data, or internal files.

---

## Configuration Best Practices

* Use `.env` files for secrets, and **never commit them**.
* Add `.env` to `.gitignore`:

  ```gitignore
  # Environment variables
  .env
  ```
* Store configuration defaults safely in `.env.example`:

  ```env
  DATABASE_URL=
  API_KEY=
  ```

---

## Recommended Practices

* Use clear and concise commit messages — they should describe **what** changed and **why**.
* Make pull requests small and focused on a single concern.
* Run tests before committing (if applicable).
* Review code for any potential data leaks before pushing.

---

## Summary

| Rule                  | Description                                      |
| --------------------- | ------------------------------------------------ |
| **Commit format**     | `fix: summary or message`                        |
| **Branch format**     | `fix/subject` or `feat/subject`                  |
| **Git history**       | Keep linear (use rebase)                         |
| **Sensitive data**    | Never commit DB creds, API keys, or private info |
| **Environment files** | Use `.env` locally, `.env.example` for structure |