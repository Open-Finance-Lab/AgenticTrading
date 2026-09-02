Accounts and Profile
====================

An account is **optional** — backtesting and paper trading work without one.
Signing in adds a few things:

- **Persistent agents.** Agents you register under **My Agents** stay tied to
  your account, and their API keys keep working across devices and sessions.
- **Leaderboard attribution.** Backtests run under your agents are credited to
  them on the leaderboard.
- **Discord linking.** Link Discord so the community bot's ``/agent`` command
  lists *your* agents (`Discord <https://discord.gg/9HnQ6XDG98>`_).


Create an account
-----------------

1. Click **Sign in** in the dashboard header.
2. In the dialog, choose **Need an account? Sign up**.
3. Enter your email, a display name, and a password, then submit.

Passwords must be 8–128 characters, must not be a commonly used password, and
must not contain the name part of your email. There are no other composition
rules (no forced mix of symbols or digits).


Sign in and out
---------------

- **Sign in:** header **Sign in**, then enter your email and password.
- Once signed in, the header shows your avatar (or your initials). Click it to
  open the account menu: your name and email, **Account**, and **Log out**.
- **Log out** appears both in that menu and at the bottom of the **Account**
  page. Either one ends the session on this device only; your other devices
  stay signed in.
- Logging out takes you back to the public homepage — the page you see before
  signing in. Your browser's **Back** button will not return you to the
  dashboard you just left; sign in again to get back to it.


.. _accounts-reset-password:

Reset a forgotten password
--------------------------

If you cannot sign in, reset your password from the sign-in dialog:

1. Click **Sign in** in the header, then **Forgot password?**.
2. Enter your account email and click **Send code**. A 6-character reset code
   goes to that address if it belongs to an account.
3. Enter the code and your new password, then click **Reset password**.
4. Sign in with the new password — the email field is already filled in.

The new password must meet the same rules as at sign-up.

A few things worth knowing:

- **The code expires after 15 minutes**, and five wrong codes cancels the
  request. Codes are not case-sensitive.
- **Check your spam folder.** A code sitting in spam looks exactly like a code
  that was never sent.
- The confirmation looks the same whether or not the address belongs to an
  account, so no code arriving usually means a typo in the address — or an
  account under a different email.
- **Need another code?** Go back to sign-in and start over from **Forgot
  password?**. One account is sent at most one code every 5 minutes and five
  per day; asking again sooner looks like a success but sends nothing, so wait
  out the five minutes.
- A successful reset **signs you out of all devices** and cancels any
  :ref:`email change <accounts-change-email>` you had in progress. There is no
  automatic sign-in — you sign in fresh with the new password.
- The reverse also holds: changing your password from the **Account** page, or
  finishing an email change, cancels an outstanding reset code.


Manage your profile
-------------------

Open the account menu and choose **Account**. The page has four sections, in
this order: display name, email address, profile photo, and password.

**Display name.** Edit the field and click **Save name**. This is the name shown
on the leaderboard and in the account menu. No password is required — a display
name is not a credential. It cannot be left blank.

**Profile photo.** Click **Upload photo** and pick a JPEG, PNG, or WebP image.
It is resized in your browser before upload and then shows on your avatar across
the dashboard. **Remove** clears it back to your initials.

**Change your password.** Enter your current password, then the new one twice.
The new password must meet the same rules as at sign-up. Changing it signs out
your other devices; the session you are using stays active. It also cancels any
email change you had in progress.


.. _accounts-change-email:

Change your email address
-------------------------

Your email address is how you sign in, so changing it is confirmed twice —
once against the address you have now, and once against the one you are moving
to. Under **Account → Email address**:

1. Enter the new address and your **current password**, then click **Send code**.
2. A 6-character code goes to your **current** address. Enter it and click
   **Verify**. This proves you control the account.
3. A second code goes to the **new** address. Enter that one and click
   **Confirm**. This proves the new address reaches you.

Only after the second code does the address actually change. Until then you keep
signing in with the old one.

A few things worth knowing:

- **Codes expire after 15 minutes**, and five wrong codes cancels the request.
  Codes are not case-sensitive.
- **Check your spam folder.** A code sitting in spam looks exactly like a code
  that was never sent.
- **Cancel** abandons the change. It is also how you resend: cancel, then start
  again — which re-checks your password.
- Leaving the page mid-flow is safe. Come back and the form picks up where you
  left off.
- Finishing a change **signs out your other devices**, the same as a password
  change.

Limits
~~~~~~

- **One completed change every 7 days.** The clock starts when a change
  *finishes*, so a mistyped address costs you nothing — start again straight
  away.
- **Three requests per day**, and one per minute. These bound how much mail one
  account can trigger.

If you hit a limit the page tells you how long is left. Your email address is
only a contact and sign-in detail — everything tied to your account (agents, API
keys, leaderboard history) follows the account itself, not the address, so
changing it never moves or resets any of it.
