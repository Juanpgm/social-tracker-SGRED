"""The local web UI: a server-rendered source manager for one operator on this machine.

It reuses the services from `social_tracker.bootstrap` (never the storage adapters), binds to the
loopback interface by default and makes no outbound network call. See `security` for the
protections every request goes through.
"""
