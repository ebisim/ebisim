"""
Exposes the dash server for easy deployment on openshift
"""
import dashboard

#Expose the server object for gunicorn
application = dashboard.app.server