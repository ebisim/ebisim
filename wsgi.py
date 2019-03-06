"""
Exposes the dash server for easy deployment on openshift
"""
import dashboard

#Fix the routing for dash
dashboard.app.config.update({
    'routes_pathname_prefix': 'ebisim/',

    'requests_pathname_prefix': 'ebisim/'
})

#Expose the server object for gunicorn
application = dashboard.app.server