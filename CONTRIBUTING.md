# Contributing Guide

Welcome to our project! Since we are a small university team, we use a very simple and lightweight workflow to make sure code is reviewed before it gets merged into `main`.

## 1. Branching
When starting a new task, create a branch from `main`:
```bash
git checkout main
git pull
git checkout -b your-feature-name
```

## 2. Opening a Merge Request (MR)
When your feature is done, push your branch and open an MR on GitLab.
- The default MR template will load automatically.
- Fill out the template explaining what you did and how to test it.
- **Do not merge your own code immediately.**

## 3. Peer Review & Approval
Because GitLab Free does not have an "Enforce Approvals" setting, we use a custom CI workaround.
When you open an MR, the pipeline will run tests and then **stop** on a manual job called `peer-approval`.

**For the Reviewer:**
1. Read the code changes in the MR.
2. If everything looks good, go to the **Pipelines** tab of the MR.
3. Find the `peer-approval` job and click the **Play (▶️)** button.
4. The pipeline will now succeed, and the MR can be merged!

## 4. Merging
Once the pipeline has succeeded (because your reviewer clicked Play), you can click **Merge** to bring your changes into `main`.
