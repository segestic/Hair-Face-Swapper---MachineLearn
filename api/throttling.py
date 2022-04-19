from rest_framework.throttling import UserRateThrottle
from rest_framework.throttling import AnonRateThrottle

class CustomUserRateThrottle(UserRateThrottle):
    scope = 'custom_user'
    rate= '5/day' # rate = '3/sec' # rate = '6/min' #'2/hour'

#new
ANON = 4
class CustomAnonRateThrottle(AnonRateThrottle):
    scope = 'custom_anon'
    #be careful to give anonymous users perission to delete..
    def allow_request(self, request, view):
        if request.method in ("GET", "POST", "PUT", "PATCH", "DELETE"):  
            self.duration = 1000#OPTIONAL - overiding the defult 86400   i.e  request available in next 10490 Seconds         
            self.num_requests = ANON
        return super().allow_request(request, view)    
#end-NEWWWWWW    
 
 


