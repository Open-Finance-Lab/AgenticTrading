.. _credits-billing:

Credits & Billing
=================

A backtest driven by a language model makes one model call per decision, and
those calls cost money. Before an LLM backtest can start you sign in and choose
who pays for them: **your own API key**, or **ATL Credits**. Rule-based
backtests make no model calls, need no account and cost nothing.

Open the page from the account menu: click your avatar, then **Credits &
Billing**. It has three tabs: **API Keys**, **Credits** and **Activity**.


Choose how AI calls are paid for
--------------------------------

The **Run Backtest** dialog for an LLM agent has an **AI billing** switch with
two options.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - **Use my API key** (BYOK)
     - **Use ATL Credits**
   * - Who pays
     - Your model provider bills your own account directly.
     - Your ATL Credit balance is debited.
   * - What it needs
     - A verified key for that provider saved under **API Keys**, and a
       **Provider** chosen in the dialog.
     - A Credit balance. There is no provider to choose; ATL picks an
       available one automatically.
   * - ATL Credits spent
     - None. A BYOK run never touches your balance.
     - Yes, per model call (see below).

If neither option is available to you yet, the dialog offers **Go to API Keys**
so you can add and verify a key. The switch only appears for LLM-driven runs;
rule-based runs and the hosted AI Hedge Fund runtime do not show it.


Use your own API key
~~~~~~~~~~~~~~~~~~~~

On the **API Keys** tab:

1. Pick a provider, then follow its link to the provider's own key page and
   create a key there.
2. Give the key a **Name**, paste the key, and click **Save and verify**.
3. Tick **Make this the verified default** if it should be the key used for that
   provider.

ATL checks the key with the provider before it counts as usable. A saved key
shows as **Verified**, **Invalid**, **Revoked** or **Verification unavailable**.
Keys stay with your account, are stored encrypted, and are never shown again
after you save them. If a key fails to verify, check that the whole key was
copied and is still active; some providers also need billing set up on your
account with them before API calls will run. You can **Reverify** a key, make it the
default (**Set default**), or **Delete** it from the list.

A BYOK run is billed entirely by your provider, so a provider-side problem (an
empty account, a rate limit) stops the run on their side, not ours.


Use ATL Credits
~~~~~~~~~~~~~~~

Credits are the unit ATL uses to pay for model calls on your behalf. They are
**not** trading capital: buying Credits never changes the simulated cash an
agent trades with, and Credits cannot be withdrawn.

The conversion is fixed: **$1 = 1 Credit**. Balances are shown to six decimal
places, for example ``1.500000 Credits``, because a single model call costs a
small fraction of a Credit.

Billing is by use, not by run. For every model call:

1. Before the call, ATL puts a hold on your balance for the most that call
   could cost. The hold is not a charge.
2. When the model answers, ATL charges what the provider reports it actually
   used (input and output tokens, at that model's price) and releases the rest
   of the hold.
3. If the call fails and produces nothing, the whole hold is released and you
   pay nothing for it.

So a run costs the sum of its calls: a longer window or a larger pipeline makes
more calls, and more assets make each call larger, so both cost more, and a short run costs very little. If a
run ends early for any reason, holds that were never settled are released too.

Welcome Credits are spent before purchased Credits.


Welcome Credits
---------------

Every account gets a one-time welcome grant of **1.5 Credits** to try LLM
backtests. It is added when you sign up, and the Activity tab lists it as
**Welcome Credits**. If it could not be applied at sign-up, it is retried the
next time you sign in. It is granted once per account.


Buy Credits (Stripe Test Mode)
------------------------------

Buying Credits runs in **Stripe Test Mode only**. **No real money moves**: you
pay with Stripe's test cards and nothing is charged. The Credits page carries a
**Test Mode** badge to say so.

An operator has to enable Test Mode billing on the server for purchases to
work. When it is not enabled the page says *Stripe Test Mode is not configured
on this server* and purchases are disabled; everything else, including BYOK and
your balance, keeps working.

To add Credits:

1. Open **Credits & Billing**, then the **Credits** tab, signed in.
2. Under **Choose an amount**, pick a package (**$0.50**, **$1**, **$2** or
   **$5**) or type a **Custom amount** from **$0.50 to $5.00**.
3. Click **Continue to Stripe** and pay on the Stripe page with a test card, for
   example ``4242 4242 4242 4242`` with any future expiry, any three-digit CVC
   and any postal code. Never enter a real card.
4. You return to ATL. The page first shows *Waiting for Stripe payment
   confirmation…* and then *Payment confirmed. Credits are now available.*

Credits are added only once Stripe's confirmation reaches ATL, not when you
return to the page, so a brief wait is normal. If the page still says
*Payment confirmation pending*, refresh it in a moment. If you close the Stripe
page without paying, or the payment does not complete, no Credits are added.

There is no self-service refund. Refunds of test purchases are handled by an
administrator.


Balance and Activity
--------------------

The **Credits** tab shows your **Credit balance**: what you have available to
spend, with the account status underneath it. The balance does not include the
holds of calls in flight.

The **Activity** tab is the ledger behind that number, newest first (the 50 most
recent entries). Each entry has a title, a timestamp and an amount:

- **Welcome Credits** — the one-time grant (``+``).
- **Credit purchase** — a Test Mode purchase (``+``).
- **Backtest usage** — what a run's model calls cost. It names the provider and
  model (or *Multiple providers* / *Multiple models*), how many model calls the
  run made, and the run it belongs to.
- **Refund** — a refund of a purchase.

BYOK runs never appear here, because they cost no Credits. Their cost is on your
provider's own dashboard.


When Credits run out, or a provider fails
-----------------------------------------

- **Not enough balance.** Each call needs enough available balance to cover its
  hold. If you are short, the model call cannot be billed and the run stops with
  *Model usage billing could not be completed*. Buy more Credits and run again,
  or switch the run to **Use my API key**.
- **Account paused.** In rare cases a call costs more than its hold and more
  than your remaining balance. The difference is recorded as an unpaid amount
  and the account is paused until it is covered. The Credits tab then says how
  much to add (*Add at least … Credits to restore access*); buying Credits at
  least that large restores access. Purchases stay available while paused for
  this reason. An account can also be paused while an administrator reviews a
  payment refund; that needs an administrator to restore.
- **A provider fails.** With ATL Credits, if one provider times out, is
  unavailable or has run out of balance on ATL's side, ATL tries the next
  available provider automatically. A failed attempt is not charged. If every
  provider fails, the run ends with the provider's error, for example *The
  selected model provider is unavailable* or *has insufficient balance or
  quota*, and you are not charged for the failed calls. (An optional
  post-trade review step that fails is skipped instead and the run carries on.) This is a platform
  problem, not a sign your Credits are gone.
- **With your own key**, ATL does not switch providers: a failure from your
  provider (an invalid or revoked key, an empty account, a timeout) ends the
  run. Fix the key or the provider account, then run again.

See :doc:`accounts` for signing in and the account menu, and
:doc:`getting_started` for running a backtest.
